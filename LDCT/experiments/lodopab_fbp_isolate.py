"""
隔离实验：合成路径能重建、真实路径不能，差异到底在哪一步。

已知
------------------------------------------------------------------
  A. iradon(radon(gt))              vs gt : corr = 0.998
  B. iradon(obs.T)                  vs gt : corr = 0.016
  C. corr( radon(gt)*PIX_M , obs.T )      = 0.970

A 与 B 应当接近（因为 C 很高），但没有。本脚本把中间的每一步都单独测一遍，
用**逐步逼近**找到断点：如果 B 失败而某一步成功，差异就在那一步。

关键怀疑：`iradon` 在 output_size < 探测器数时会对 sinogram 做
**order=0 最近邻重采样**。512→362 与 513→362 的采样点若不同，
重建就会一个成一个不成。

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_isolate.py
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
from skimage.transform import iradon, radon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LODO = os.path.join(ROOT, "data", "extracted")
PIX_M = 0.000718


def corr(a, b):
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
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
        obs0 = f["data"][0].astype(np.float64)

    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)

    def rec_of(sino, tag):
        r = iradon(sino, theta=ang, output_size=362,
                   filter_name="ramp", circle=False)
        print(f"  {tag:<46}{sino.shape}  corr={corr(r, gt):>8.4f}")
        return r

    print("【逐级逼近】")
    syn = radon(gt, theta=ang, circle=False)
    r_syn = rec_of(syn, "1. iradon(radon(gt))")
    r_syn_s = rec_of(syn * PIX_M, "2. iradon(radon(gt) * PIX_M)  [模拟真实尺度]")

    # 人为把 512 bin 补成 513，看 bin 数是否是断点
    syn513_end = np.pad(syn, ((0, 1), (0, 0)))
    r_513e = rec_of(syn513_end, "3. iradon(pad_末尾 → 513 bin)")
    syn513_beg = np.pad(syn, ((1, 0), (0, 0)))
    r_513b = rec_of(syn513_beg, "4. iradon(pad_开头 → 513 bin)")
    r_513c = rec_of(np.pad(syn, ((0, 0), (0, 0)))[0:512], "5. 原样 512（对照）")

    r_real = rec_of(obs0.T, "6. iradon(obs.T)                [真实]")

    print("\n【交叉相关】各重建之间以及和 gt/合成重建的关系")
    print(f"  corr(真实重建, 合成重建)      = {corr(r_real, r_syn):.4f}")
    print(f"  corr(真实重建, gt)            = {corr(r_real, gt):.4f}")
    print(f"  corr(合成重建, gt)            = {corr(r_syn, gt):.4f}")
    print(f"  corr(513补末尾重建, 合成重建)  = {corr(r_513e, r_syn):.4f}")
    print(f"  corr(513补开头重建, 合成重建)  = {corr(r_513b, r_syn):.4f}")

    print("\n【直接看 sinogram 的逐列结构】")
    print("  取中间 3 个角度，比较合成与真实的投影剖面形状")
    for i in (300, 500, 700):
        a = syn[6:506, i]
        b = obs0[i, 1:501]
        c = obs0[i, 0:500]
        print(f"    角度 {i}: corr(syn, obs[1:501])={corr(a, b):+.4f}   "
              f"corr(syn, obs[0:500])={corr(a, c):+.4f}")

    print("\n【判读】")
    print("  · 若 1 好、2 坏  => 问题在尺度（不该，iradon 是线性的）")
    print("  · 若 3/4 坏       => bin 数 512 vs 513 确实是断点")
    print("  · 若逐列 corr 高但整体重建坏 => 角度轴的对应关系不对")


if __name__ == "__main__":
    main()
