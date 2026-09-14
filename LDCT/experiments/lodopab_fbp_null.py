"""
补上对照组：先前"逐角度投影剖面相关 0.92–0.97"很可能是**假阳性**。

为什么必须加对照
------------------------------------------------------------------
胸部 CT 在任何角度下的投影都是一条相似的宽钟形曲线。因此
`corr(syn[:,i], obs[i,:])` 天然就高，**不能证明两者对上了**。
正确做法是同时算一个 null：拿**不同角度**的剖面互相比。
若 null 也有 0.9，则这个指标毫无判别力。

本脚本：
  1. 打印剖面的实际数值（每隔若干点），肉眼比对是否存在平移/缩放
  2. 算 null 对照：corr(syn[:,i], obs[j,:]) for i≠j
  3. 找到 obs 的每个角度**真正对应** syn 的哪个角度（最大相关）
     —— 若存在系统性偏移或次序反转，这里会直接看出来

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_null.py
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
from skimage.transform import radon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LODO = os.path.join(ROOT, "data", "extracted")
PIX_M = 0.000718


def corr(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    with h5py.File(os.path.join(LODO, "ground_truth_test_000.hdf5"), "r") as f:
        gt = f["data"][0].astype(np.float64)
    with h5py.File(os.path.join(LODO, "observation_test_000.hdf5"), "r") as f:
        obs = f["data"][0].astype(np.float64)

    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)
    syn = radon(gt, theta=ang, circle=False) * PIX_M      # (512, 1000)

    # ---- 1 剖面数值直接打印 ----
    print("【1】角度 500 的投影剖面，逐点数值对比")
    print(f"  {'idx':>5}{'syn[6+idx]':>14}{'obs[500, 1+idx]':>18}{'obs[500, idx]':>16}")
    for k in range(0, 500, 50):
        print(f"  {k:>5}{syn[6+k, 500]:>14.5f}{obs[500, 1+k]:>18.5f}"
              f"{obs[500, k]:>16.5f}")

    # ---- 2 null 对照 ----
    print("\n【2】null 对照：同一剖面 vs 不同角度")
    a = syn[6:506, 500]
    print(f"  corr(syn[:,500], obs[500, 1:501]) = {corr(a, obs[500, 1:501]):+.4f}   <- 同角度")
    for j in (100, 300, 700, 900):
        print(f"  corr(syn[:,500], obs[{j}, 1:501]) = {corr(a, obs[j, 1:501]):+.4f}"
              f"   <- 不同角度（null）")
    print("  若 null 也有 0.9 量级 => 该指标无判别力，先前的 0.92–0.97 不作数")

    # ---- 3 每个 obs 角度真正对应 syn 的哪个角度 ----
    print("\n【3】为若干 obs 角度找最佳匹配的 syn 角度")
    print(f"  {'obs角度':>8}{'最佳syn角度':>13}{'最大corr':>11}{'原始corr':>11}")
    for i in (0, 125, 250, 375, 500, 625, 750, 875):
        b = obs[i, 1:501]
        cs = np.array([corr(syn[6:506, j], b) for j in range(0, 1000, 10)])
        jbest = int(np.argmax(cs)) * 10
        print(f"  {i:>8}{jbest:>13}{cs.max():>11.4f}"
              f"{corr(syn[6:506, i], b):>11.4f}")

    # ---- 4 整体：obs 的角度轴与行列结构 ----
    print("\n【4】obs 与 syn 的整体相似度矩阵（粗采样）")
    idx = list(range(0, 1000, 100))
    print("        " + "".join(f"{j:>8}" for j in idx))
    for i in idx:
        row = []
        for j in idx:
            row.append(corr(syn[6:506, j], obs[i, 1:501]))
        print(f"  {i:>5} " + "".join(f"{v:>8.3f}" for v in row))
    print("  理想情况：对角线附近应为最高；若整体错位则峰值出现在别处")


if __name__ == "__main__":
    main()
