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
- 实测 HU 范围约 [-1024, 1185]

⚠️ 配对必须按 **ImagePositionPatient[2]（z 位置）**，不能按文件名。
   实测两侧文件名的第 5 段都是 "12"，无法区分；而 z 位置严格一一对应。

口径约定（**写进论文时必须说明**）
------------------------------------------------------------------
1. **HU 变换**：`HU = pixel_array * RescaleSlope + RescaleIntercept`
2. **窗宽窗位**：默认裁剪到 `[-1000, 1000]` HU。
   ⚠️ **这是本脚本最需要外部确认的一项**：不同论文用的窗可能不同
   （常见还有 [-1024, 3071] 全范围、[-1000, 1500] 等）。
   窗不同会导致 PSNR 不可比。**与已发表数字对比前，务必先核对对方的口径。**
   参数 `--hu-min/--hu-max` 可覆盖。
3. **归一化**：`x = (clip(HU, lo, hi) - lo) / (hi - lo)` → [0, 1]。
   固定窗而非逐图 min-max——逐图归一化会随图像内容变化，破坏可比性。
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


def _read_series(dirpath: str, hu_min: float, hu_max: float):
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
        hu = np.clip(hu, hu_min, hu_max)
        x = (hu - hu_min) / (hu_max - hu_min)
        zs.append(z)
        slices.append(x.astype(np.float32))
    return zs, np.stack(slices)


def build(src: str, out: str, hu_min: float, hu_max: float):
    patients = sorted(d for d in os.listdir(os.path.join(src, "full_3mm"))
                      if os.path.isdir(os.path.join(src, "full_3mm", d)))
    if not patients:
        raise FileNotFoundError(f"{src}/full_3mm 下没有患者目录")

    os.makedirs(out, exist_ok=True)
    index = {"created": datetime.now(timezone.utc).isoformat(),
             "source": os.path.relpath(src, ROOT),
             "hu_window": [hu_min, hu_max],
             "normalization": "(clip(HU,lo,hi)-lo)/(hi-lo) -> [0,1]",
             "pairing": "按 ImagePositionPatient[2] 排序后一一对应",
             "patients": {}}

    for pid in patients:
        fd = os.path.join(src, "full_3mm", pid, "full_3mm")
        qd = os.path.join(src, "quarter_3mm", pid, "quarter_3mm")
        zf, Xf = _read_series(fd, hu_min, hu_max)
        zq, Xq = _read_series(qd, hu_min, hu_max)

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
    print(f"HU 窗: [{hu_min:.0f}, {hu_max:.0f}]")
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
    ap.add_argument("--hu-min", type=float, default=-1000.0)
    ap.add_argument("--hu-max", type=float, default=1000.0)
    ap.add_argument("--test-patients", default="L506,L067")
    ap.add_argument("--val-patients", default="L291")
    args = ap.parse_args()

    print("=" * 74)
    print("AAPM-Mayo 2016 LDCT 数据准备")
    print(f"  源目录 : {args.src}")
    print(f"  HU 窗  : [{args.hu_min:.0f}, {args.hu_max:.0f}]")
    print("=" * 74)
    idx = build(args.src, args.out, args.hu_min, args.hu_max)
    make_splits(idx, [p for p in args.test_patients.split(",") if p],
                [p for p in args.val_patients.split(",") if p],
                os.path.join(ROOT, "splits", "aapm_mayo_3mm.json"))


if __name__ == "__main__":
    main()
