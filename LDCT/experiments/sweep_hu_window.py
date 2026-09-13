"""
扫描 HU 窗对指标的影响。

为什么必须做
------------------------------------------------------------------
PSNR 的定义依赖数据范围：

    x_norm = (clip(HU, lo, hi) − lo) / R,   R = hi − lo
    MSE_norm = MSE_HU / R²
    PSNR = −10·log10(MSE_norm) = −10·log10(MSE_HU) + 20·log10(R)

**即：窗越宽，PSNR 越高，偏移量为 20·log10(R)。**

    R = 2000 (窗 [-1000,1000])   → 20·log10(2000) = 66.0 dB
    R = 4095 (窗 [-1024,3071])   → 20·log10(4095) = 72.2 dB
    差值 = 6.2 dB

**6 dB 是巨大的差异**——足以让一个方法"看起来"从落后变成领先。
故各论文若不声明窗，其 PSNR 根本不可比。

本脚本在**同一个测试集**上，对多个候选窗计算 identity 基线（输出=输入）的
PSNR/SSIM，以及被裁剪像素的比例。用途：

  1. 量化"窗的选择"本身能造成多大差异
  2. 提供一个换算表——若知道某论文的窗，可估出我们的数字在该窗下会变成多少

⚠️ 裁剪会改变信号本身（不只改变量纲），故窗过窄时 PSNR 不会简单按公式涨。

用法：
    python experiments/sweep_hu_window.py --test-patients L506,L067
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pydicom

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AAPM = os.path.join(ROOT, "data", "aapm")

# 候选窗：(名称, lo, hi)
CANDIDATES = [
    ("[-1000, 1000]  软组织+骨（本项目当前）", -1000.0, 1000.0),
    ("[-1024, 3071]  CT 理论全范围", -1024.0, 3071.0),
    ("[-1024, 2048]  常见全范围变体", -1024.0, 2048.0),
    ("[-1000, 2000]  偏宽", -1000.0, 2000.0),
    ("[-1024, 1500]  文献常见", -1024.0, 1500.0),
    ("[-1024, 1185]  本数据实际范围", -1024.0, 1185.0),
    ("[-160, 240]    腹部软组织窗", -160.0, 240.0),
    ("[-1000, 400]   腹部+部分骨", -1000.0, 400.0),
]


def read_patient_hu(pid: str, dose: str = "quarter"):
    """读一个患者的全部切片，返回**原始 HU**（未裁剪）的数组栈。"""
    d = os.path.join(AAPM, f"{dose}_3mm", pid, f"{dose}_3mm")
    files = glob.glob(os.path.join(d, "*.IMA"))
    entries = []
    for f in files:
        h = pydicom.dcmread(f, stop_before_pixels=True)
        entries.append((float(h.ImagePositionPatient[2]), f))
    entries.sort(key=lambda t: t[0])
    out = []
    for _, f in entries:
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        hu = arr * float(getattr(ds, "RescaleSlope", 1.0)) + \
             float(getattr(ds, "RescaleIntercept", 0.0))
        out.append(hu)
    return np.stack(out)


def psnr_ssim_on_window(x_obs_hu, x_gt_hu, lo, hi):
    """在给定窗下归一化后，算 identity 基线（输出=输入）的 PSNR/SSIM。"""
    from skimage.metrics import structural_similarity

    o = np.clip(x_obs_hu, lo, hi)
    g = np.clip(x_gt_hu, lo, hi)
    R = float(hi - lo)
    o = (o - lo) / R
    g = (g - lo) / R

    mse = float(np.mean((o - g) ** 2))
    psnr = float("inf") if mse == 0 else float(10.0 * np.log10(1.0 / mse))

    # SSIM 逐切片算再平均（data_range=1）
    ss = []
    for k in range(o.shape[0]):
        ss.append(structural_similarity(g[k], o[k], data_range=1.0,
                                        gaussian_weights=True, sigma=1.5,
                                        use_sample_covariance=False, win_size=11))
    clipped = float(np.mean((x_obs_hu < lo) | (x_obs_hu > hi)))
    return psnr, float(np.mean(ss)), clipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-patients", default="L506,L067")
    ap.add_argument("--out", default=os.path.join(HERE, "hu_window_sweep.json"))
    args = ap.parse_args()

    pids = [p for p in args.test_patients.split(",") if p]
    print("=" * 96)
    print(f"HU 窗扫描 —— identity 基线（输出=输入），测试患者 {pids}")
    print("=" * 96)

    data = {}
    for pid in pids:
        print(f"  读取 {pid} ...", end="", flush=True)
        data[pid] = (read_patient_hu(pid, "quarter"),
                     read_patient_hu(pid, "full"))
        print(f" {data[pid][0].shape[0]} 片")

    print()
    print(f"  {'HU 窗':<38}{'范围R':>8}{'identity PSNR':>15}{'SSIM':>10}{'裁剪%':>9}")
    print("  " + "-" * 88)

    rows = []
    for name, lo, hi in CANDIDATES:
        ps, ss, cl = [], [], []
        for pid in pids:
            o, g = data[pid]
            p, s, c = psnr_ssim_on_window(o, g, lo, hi)
            ps.append(p); ss.append(s); cl.append(c)
        # 按切片加权（患者切片数不同）
        w = np.array([data[p][0].shape[0] for p in pids], dtype=float)
        w = w / w.sum()
        P, S, C = float(np.average(ps, weights=w)), float(np.average(ss, weights=w)), \
                  float(np.average(cl, weights=w))
        rows.append({"window": name, "lo": lo, "hi": hi, "R": hi - lo,
                     "psnr": P, "ssim": S, "clipped_frac": C})
        print(f"  {name:<38}{hi-lo:>8.0f}{P:>15.4f}{S:>10.4f}{C*100:>8.2f}%")

    print()
    base = next(r for r in rows if r["lo"] == -1000.0 and r["hi"] == 1000.0)
    print("  相对窗 [-1000,1000] 的 PSNR 偏移（= 20·log10(R 之比)，裁剪不重时成立）：")
    for r in rows:
        d_psnr = r["psnr"] - base["psnr"]
        theo = 20 * np.log10(r["R"] / base["R"])
        print(f"    {r['window']:<38} 实测 {d_psnr:+7.3f} dB   理论 {theo:+7.3f} dB")
    print()
    print("  ⚠️ 实测与理论不符的行说明该窗发生了显著裁剪，信号被改变。")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"test_patients": pids, "rows": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
