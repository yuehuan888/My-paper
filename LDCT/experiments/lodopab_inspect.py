"""
LoDoPaB-CT 数据勘察：在决定评测口径之前，先看清它到底是什么。

为什么必须先做这一步
------------------------------------------------------------------
零样本跨库测试最容易在这里翻车：模型在 AAPM 上训练时，输入是
`(HU + 1024) / 4096`（HU=-1024 → 0，HU=3072 → 1）。
若 LoDoPaB 的 [0,1] 是**另一套缩放**，把它的图直接喂进去就是分布外输入，
跑出来的数字没有任何意义——**而且不会报错，只会静默地给出一个很差的分数**。

所以本脚本只做测量，不做假设：
  · 形状、dtype、逐片 vs 分片
  · 值域与分位数
  · gt 与 observation 的关系（同一缩放？还是差一个增益？）
  · 与 AAPM 的分布对比

用法
------------------------------------------------------------------
    python experiments/lodopab_inspect.py
    python experiments/lodopab_inspect.py --n-shards 2
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

LODO_ROOT = os.path.join(ROOT, "data", "extracted")
AAPM_H5 = os.path.join(ROOT, "data", "aapm_h5")


def describe(name, a):
    a = np.asarray(a, dtype=np.float64)
    qs = np.percentile(a, [0, 0.1, 1, 50, 99, 99.9, 100])
    print(f"    {name:<14} shape={str(a.shape):<18} dtype={a.dtype} "
          f"mean={a.mean():+.4f} std={a.std():.4f}")
    print(f"      min/0.1%/1%/中位/99%/99.9%/max = "
          + "  ".join(f"{q:+.4f}" for q in qs))
    print(f"      负值占比 {np.mean(a < 0)*100:.4f}%   >1 占比 {np.mean(a > 1)*100:.4f}%")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-shards", type=int, default=2,
                    help="勘查询多少个分片（每片约 260 MB，默认 2）")
    args = ap.parse_args()

    if not os.path.isdir(LODO_ROOT):
        print(f"❌ {LODO_ROOT} 不存在——数据还没解压。")
        return 1

    gt_files = sorted(f for f in os.listdir(LODO_ROOT)
                      if f.startswith("ground_truth_test_") and f.endswith(".hdf5"))
    ob_files = sorted(f for f in os.listdir(LODO_ROOT)
                      if f.startswith("observation_test_") and f.endswith(".hdf5"))
    print("=" * 78)
    print(f"LoDoPaB-CT  test 分片：ground_truth {len(gt_files)} 个，"
          f"observation {len(ob_files)} 个")
    print("=" * 78)

    n = min(args.n_shards, len(gt_files), len(ob_files))
    total_slices = 0

    for i in range(n):
        gp = os.path.join(LODO_ROOT, gt_files[i])
        op = os.path.join(LODO_ROOT, ob_files[i])
        print(f"\n--- 分片 {i} ---")
        with h5py.File(gp, "r") as f:
            print(f"  {gt_files[i]}  键={list(f.keys())}")
            for k in f.keys():
                print(f"     /{k}: shape={f[k].shape} dtype={f[k].dtype} "
                      f"attrs={dict(f[k].attrs)}")
            g = f["data"][: min(8, f["data"].shape[0])]
        with h5py.File(op, "r") as f:
            print(f"  {ob_files[i]}  键={list(f.keys())}")
            for k in f.keys():
                print(f"     /{k}: shape={f[k].shape} dtype={f[k].dtype} "
                      f"attrs={dict(f[k].attrs)}")
            o = f["data"][: min(8, f["data"].shape[0])]

        if i == 0:
            with h5py.File(gp, "r") as f:
                total_slices = f["data"].shape[0]
            print(f"\n  每个分片 {total_slices} 张切片 "
                  f"→ test 共 {total_slices * len(gt_files)} 张")

        print()
        describe("ground_truth", g)
        describe("observation", o)

        g64, o64 = g.astype(np.float64), o.astype(np.float64)
        m = g64.mean()
        if m > 1e-9:
            print(f"\n      observation / ground_truth 均值比 = {o64.mean()/m:.6f}")
        d = (o64 - g64)
        print(f"      obs-gt 差：mean={d.mean():+.5f} std={d.std():.5f}")
        # 线性拟合 obs ≈ a*gt + b，判断是否只差一个仿射
        a, b = np.polyfit(g64.ravel(), o64.ravel(), 1)
        pred = a * g64 + b
        resid = np.abs(pred - o64).mean()
        print(f"      线性拟合 obs ≈ {a:.6f}·gt {b:+.6f}，"
              f"拟合残差均值 {resid:.6f}（相对 {d.std():.6f}）")

    # ------------------------------------------------ 与 AAPM 对比
    print("\n" + "=" * 78)
    print("与 AAPM-Mayo 的分布对比（AAPM 的输入是 (HU+1024)/4096）")
    print("=" * 78)
    if os.path.isdir(AAPM_H5):
        h5s = sorted(f for f in os.listdir(AAPM_H5) if f.endswith(".h5"))
        if h5s:
            with h5py.File(os.path.join(AAPM_H5, h5s[0]), "r") as f:
                print(f"\n  AAPM {h5s[0]}  键={list(f.keys())}")
                gg = f["ground_truth"][:8]
                oo = f["observation"][:8]
            describe("AAPM gt", gg)
            describe("AAPM obs", oo)
            print(f"\n  ⚠️ 对比要点：若 LoDoPaB 的中位数/分位数与 AAPM 差得远，")
            print(f"     说明两者的 [0,1] 不是同一套缩放，**不能直接喂给模型**。")
    else:
        print(f"  （未找到 {AAPM_H5}）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
