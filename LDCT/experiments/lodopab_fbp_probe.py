"""
LoDoPaB 能不能用？先验证 FBP 重建这条路通不通。

背景
------------------------------------------------------------------
2026-09-14 实测确认：LoDoPaB-CT 是**重建**基准，不是去噪基准。
  ground_truth  (N, 362, 362)  ← 重建后的图像
  observation   (N, 1000, 513) ← **sinogram**（1000 角度 × 513 探测器）
Zenodo 官方描述原文："A Benchmark Dataset for Low-Dose CT Reconstruction Methods."
且全站没有任何图像域的低剂量版本。

所以"零样本去噪"必须先自己 FBP 重建。本脚本验证这条路是否可行：

  1. 用 skimage 的 iradon 重建，看输出值域是否合理
  2. 与 ground_truth 做仿射拟合，看残差（拟合得好 => 几何参数对）
  3. 报告重建图的 PSNR/SSIM —— 这是"含噪输入"相对于真值的水平，
     也是模型必须超过的地板

几何参数（需要验证）
------------------------------------------------------------------
  图像 362×362，探测器 513 像元。并行束要覆盖 362×362 图的对角线
  （362·√2 ≈ 512），故 513 像元 ≈ 512+1，**探测器间距 ≈ 1 个像素**，
  正好是 skimage iradon 的默认假设。角度 1000 个，均匀覆盖 [0, 180°)。

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_probe.py --n 8
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
from skimage.transform import iradon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LODO = os.path.join(ROOT, "data", "extracted")
RUNS = os.path.join(HERE, "runs")


def to_hu_norm(x):
    """与 utils.metrics 的口径 A 对齐：x 是 [0,1] 域，转回 HU 便于对比。"""
    return x * 4096.0 - 1024.0


def psnr(a, b, data_range):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(data_range ** 2 / mse)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="试几张")
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    gp = os.path.join(LODO, f"ground_truth_test_{args.shard:03d}.hdf5")
    op = os.path.join(LODO, f"observation_test_{args.shard:03d}.hdf5")
    with h5py.File(gp, "r") as f:
        gt = f["data"][:args.n].astype(np.float32)
    with h5py.File(op, "r") as f:
        obs = f["data"][:args.n].astype(np.float32)

    print(f"gt  {gt.shape}  值域 [{gt.min():.4f}, {gt.max():.4f}]")
    print(f"obs {obs.shape}  值域 [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"角度数 {obs.shape[1]}，探测器像元 {obs.shape[2]}\n")

    angles = np.linspace(0.0, 180.0, obs.shape[1], endpoint=False)

    print(f"  {'#':<4}{'FBP值域':>22}{'拟合 a':>10}{'拟合 b':>10}"
          f"{'拟合残差':>11}{'PSNR(拟合后)':>14}")
    print("  " + "-" * 74)

    recs = np.zeros_like(gt)
    for i in range(args.n):
        # iradon 要求 (探测器, 角度)
        sino = obs[i].T
        rec = iradon(sino, theta=angles, output_size=gt.shape[-1],
                     filter_name="ramp", circle=False)
        recs[i] = rec

        g = gt[i].astype(np.float64)
        r = rec.astype(np.float64)
        # 仿射拟合 rec ≈ a·g + b（全局尺度问题：FBP 给的是 μ，gt 是归一化值）
        A = np.stack([g.ravel(), np.ones(g.size)], 1)
        sol, *_ = np.linalg.lstsq(A, r.ravel(), rcond=None)
        a, b = sol
        resid = float(np.abs(A @ sol - r.ravel()).mean())
        p = psnr(a * g + b, r, data_range=float((a * g + b).max() - (a * g + b).min()) or 1.0)
        print(f"  {i:<4}[{rec.min():+.4f}, {rec.max():+.4f}]"
              f"{a:>10.5f}{b:>10.5f}{resid:>11.5f}{p:>14.2f}")

    # 全局尺度：用所有试片拟合一个 a,b，再看重建质量
    print("\n  --- 用全部试片拟合**单一全局** a,b（不逐图拟合，避免偷看真值）---")
    G = gt.astype(np.float64).ravel()
    R = recs.astype(np.float64).ravel()
    A = np.stack([G, np.ones(G.size)], 1)
    sol, *_ = np.linalg.lstsq(A, R, rcond=None)
    a, b = sol
    print(f"    全局 a = {a:.6f}   b = {b:+.6f}")

    ps, ss = [], []
    for i in range(args.n):
        pred = a * gt[i].astype(np.float64) + b
        r = recs[i].astype(np.float64)
        ps.append(psnr(pred, r, data_range=float(pred.max() - pred.min())))
    print(f"    逐图 PSNR(FBP vs gt) = {np.mean(ps):.2f} dB "
          f"（范围 {min(ps):.2f} – {max(ps):.2f}）")

    # 与 identity 对比：若把 FBP 结果当作"输入"，它相对 gt 的差距就是待去噪的量
    print("\n  判读：")
    print("    · 若逐图 PSNR 落在 25–35 dB，说明几何参数基本正确，FBP 路径可行")
    print("    · 若只有 10–20 dB，说明探测器间距/角度范围不对，需要标定")
    print("    · 拟合残差若远小于 obs-gt 差，也说明线性关系成立")


if __name__ == "__main__":
    main()
