"""
AAPM-Mayo 2016 LDCT 数据准备：DICOM -> HDF5。

数据集
------------------------------------------------------------------
AAPM Low Dose CT Grand Challenge (2016)，3mm B30 配对子集。
来源：https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h
  full_3mm.zip    (682 MB) 全剂量
  quarter_3mm.zip (731 MB) 四分之一剂量
10 个患者（L067 L096 L109 L143 L192 L286 L291 L310 L333 L506），
512×512，SliceThickness 3mm，PixelSpacing 0.664，层间距 2mm。

格式（**实测确认**，非推测）
------------------------------------------------------------------
- DICOM .IMA，uint16
- RescaleSlope=1, RescaleIntercept=-1024  →  HU = pixel − 1024
- 实测 HU 上界：**L506 达 3071**（>3000 HU 有 114 像素）、L067 2129、L333 1719。

  ⚠️ 本文件早先写的"实测 HU 范围约 [-1024, 1185]"是**错的**（已更正）。
  这处错误会误导窗的选择——若按 1185 当"数据实际范围"，会裁掉真实的高密度像素
  （骨/金属/床板）。`experiments/sweep_hu_window.py` 的候选表里也有同一处，
  但因该表逐项实测，数值未受影响。

⚠️ 配对必须按 **ImagePositionPatient[2]（z 位置）**，不能按文件名。
   实测两侧文件名的第 5 段都是 "12"，无法区分；而 z 位置严格一一对应。

口径约定（**写进论文时必须说明**）
------------------------------------------------------------------
1. **HU 变换**：`HU = pixel_array * RescaleSlope + RescaleIntercept`
2. **存储时不做窗裁剪**：`x = (HU + 1024) / 4096`，直接落在 [0, 1]。
   裁剪**推迟到评测阶段**（见下），这是为了与主流口径对齐。

3. **评测口径（口径 A，RED-CNN / CTformer 谱系）**：
   - 反归一化回 HU：`HU = x * 4096 - 1024`
   - 预测与参考**同时** clip 到 `[-160, 240]`（400 HU 软组织窗）
   - `PSNR = 10·log10(400² / MSE_HU)`，即 data_range=400，RMSE 单位为 HU
   - SSIM 用 skimage，data_range=400

   **依据**（2026-09-13 核查，见 `../../LDCT_HU口径核查_2026-09-13.md`）：
   - AAPM 官方**从未定义**任何 PSNR 口径（官方指标是医师阅片评分）
   - 该数据集上至少并存 5 个互不兼容的阵营，同一模型换口径可差 **20 dB**
   - 口径 A 被 SSinyu/RED-CNN、SSinyu/WGAN-VGG、wdayang/CTformer、
     EHSANet、MRED-Net、SDCNN 逐字沿用（`(HU+1024)/4096` + clip[-160,240] + data_range=400）
   - Eulig et al. (Med Phys 2024) 独立收敛到同一个 400 HU 窗（[-150,250]）
   - 且口径 A 的源头用的正是**本数据集**（10 患者 3mm 512×512 quarter/full）

   ⚠️ `[-1000, 1000]`（本项目早期用法）**不属于任何公共口径**，
   比口径 A 高约 14 dB，数字不可与文献并列。
4. **划分**：默认按患者，`--test-patients` 指定测试患者。
   常用约定是留 2 个患者做测试。同样**需与要对比的论文核对**。

用法
------------------------------------------------------------------
    python data/prepare_aapm.py --out data/aapm_h5
    python data/prepare_aapm.py --test-patients L506,L067
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import h5py
import numpy as np
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_SRC = os.path.join(HERE, "aapm")


# 口径 A 的常数
HU_OFFSET = 1024.0     # HU = x * 4096 - 1024
HU_SCALE = 4096.0
EVAL_LO, EVAL_HI = -160.0, 240.0      # 评测窗（仅用于打印，实际计算在 utils/metrics.py）


def _read_series(dirpath: str):
    """读一个患者的一个剂量序列，按 z 排序，返回 (z 列表, HU 数组栈)。"""
    files = glob.glob(os.path.join(dirpath, "*.IMA"))
    if not files:
        raise FileNotFoundError(f"{dirpath} 下没有 .IMA 文件")

    entries = []
    for f in files:
        d = pydicom.dcmread(f, stop_before_pixels=True)
        entries.append((float(d.ImagePositionPatient[2]), f))
    entries.sort(key=lambda t: t[0])

    zs, slices = [], []
    for z, f in entries:
        d = pydicom.dcmread(f)
        arr = d.pixel_array.astype(np.float32)
        slope = float(getattr(d, "RescaleSlope", 1.0))
        inter = float(getattr(d, "RescaleIntercept", 0.0))
        hu = arr * slope + inter
        # 口径 A：不裁剪，直接线性映射到 [0,1]
        #   x = (HU + 1024) / 4096
        # ⚠️ 注意是 **加** 1024。写成减号会得到 [-0.5, 0.5] 的值域，
        #    且经评测窗 clip 后几乎全部饱和成常数，PSNR 会假性地变成 inf。
        x = (hu + HU_OFFSET) / HU_SCALE
        zs.append(z)
        slices.append(x.astype(np.float32))
    return zs, np.stack(slices)


def build(src: str, out: str):
    patients = sorted(d for d in os.listdir(os.path.join(src, "full_3mm"))
                      if os.path.isdir(os.path.join(src, "full_3mm", d)))
    if not patients:
        raise FileNotFoundError(f"{src}/full_3mm 下没有患者目录")

    os.makedirs(out, exist_ok=True)
    index = {"created": datetime.now(timezone.utc).isoformat(),
             "source": os.path.relpath(src, ROOT),
             "hu_offset": HU_OFFSET, "hu_scale": HU_SCALE,
             "normalization": "x = (HU + 1024) / 4096  -> [0,1]（不裁剪）",
             "eval_protocol": "口径A: 反归一化到HU后 pred与gt同时clip到[-160,240], data_range=400",
             "pairing": "按 ImagePositionPatient[2] 排序后一一对应",
             "patients": {}}

    for pid in patients:
        fd = os.path.join(src, "full_3mm", pid, "full_3mm")
        qd = os.path.join(src, "quarter_3mm", pid, "quarter_3mm")
        zf, Xf = _read_series(fd)
        zq, Xq = _read_series(qd)

        if len(zf) != len(zq):
            raise ValueError(f"{pid}: full {len(zf)} != quarter {len(zq)}")
        if not np.allclose(zf, zq, atol=1e-3):
            bad = int(np.sum(~np.isclose(zf, zq, atol=1e-3)))
            raise ValueError(f"{pid}: z 位置不一一对应（{bad} 处不符）")

        p = os.path.join(out, f"{pid}.h5")
        with h5py.File(p, "w") as f:
            f.create_dataset("observation", data=Xq, compression="gzip", compression_opts=4)
            f.create_dataset("ground_truth", data=Xf, compression="gzip", compression_opts=4)
            f.attrs["patient_id"] = pid
            f.attrs["n_slices"] = len(zf)
            f.attrs["z_positions"] = np.asarray(zf, dtype=np.float64)

        index["patients"][pid] = {
            "n_slices": int(len(zf)),
            "shape": [int(Xf.shape[1]), int(Xf.shape[2])],
            "z_range": [float(zf[0]), float(zf[-1])],
            "h5": os.path.relpath(p, ROOT),
            "obs_range": [float(Xq.min()), float(Xq.max())],
            "gt_range": [float(Xf.min()), float(Xf.max())],
        }
        print(f"  {pid}: {len(zf):>4} 片  {Xf.shape[1]}x{Xf.shape[2]}  "
              f"({os.path.getsize(p)/1048576:.0f} MB)")

    with open(os.path.join(out, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total = sum(v["n_slices"] for v in index["patients"].values())
    print(f"\n合计 {len(patients)} 个患者 / {total} 对切片")
    print(f"归一化: x = (HU + {HU_OFFSET:.0f}) / {HU_SCALE:.0f}  （不裁剪）")
    print(f"评测口径: 口径A, clip 到 [{EVAL_LO:.0f}, {EVAL_HI:.0f}] HU, data_range=400")
    print(f"输出: {out}")
    return index


def make_splits(index: dict, test_patients: list[str], val_patients: list[str],
                out_path: str):
    ps = sorted(index["patients"])
    for p in test_patients + val_patients:
        if p not in index["patients"]:
            raise ValueError(f"患者 {p} 不在数据集中，可选: {ps}")
    train = [p for p in ps if p not in test_patients and p not in val_patients]
    if not train:
        raise ValueError("训练集为空")

    split = {
        "name": "aapm_mayo_3mm",
        "created": datetime.now(timezone.utc).isoformat(),
        "split_policy": "按**患者**划分，避免同一患者的相邻切片跨集合",
        "train_patients": train,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "n_slices": {
            "train": sum(index["patients"][p]["n_slices"] for p in train),
            "val": sum(index["patients"][p]["n_slices"] for p in val_patients),
            "test": sum(index["patients"][p]["n_slices"] for p in test_patients),
        },
        "warning": ("跨样本比较时独立单元是**患者**而非切片。"
                    "同一患者 224–330 张相邻切片高度相关。"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)
    print(f"\n划分 -> {out_path}")
    print(f"  train: {train}  ({split['n_slices']['train']} 片)")
    print(f"  val  : {val_patients}  ({split['n_slices']['val']} 片)")
    print(f"  test : {test_patients}  ({split['n_slices']['test']} 片)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.join(HERE, "aapm_h5"))

    ap.add_argument("--test-patients", default="L506,L067")
    ap.add_argument("--val-patients", default="L291")
    args = ap.parse_args()

    print("=" * 74)
    print("AAPM-Mayo 2016 LDCT 数据准备")
    print(f"  源目录 : {args.src}")
    print(f"  归一化  : x = (HU + 1024) / 4096  -> [0,1]（不裁剪）")
    print("=" * 74)
    idx = build(args.src, args.out)
    make_splits(idx, [p for p in args.test_patients.split(",") if p],
                [p for p in args.val_patients.split(",") if p],
                os.path.join(ROOT, "splits", "aapm_mayo_3mm.json"))


if __name__ == "__main__":
    main()
