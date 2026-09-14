"""
找出 LoDoPaB 真实 sinogram 与 skimage radon 的**约定差异**。

已知（lodopab_fbp_debug.py 实测）：
  · skimage radon(gt) -> iradon 的往返 corr = 0.998   => iradon 用法正确
  · 尺度关系 obs = radon_skimage(gt) × 0.000718（像素间距，米）
  · 但对真实 obs 做 FBP，corr(rec, gt) 仅 0.0158      => 约定不一致

本脚本不再猜，直接把 radon(gt) 与真实 obs 在多种变换下比对相关度，
把差异定位到「哪个轴、哪个方向、是否平移」。

用法
------------------------------------------------------------------
    python experiments/lodopab_sino_convention.py
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

PIX_M = 0.000718          # 探测器/像素间距（米），由 dival 几何推出


def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def align_rows(syn, real, off):
    """把 syn 的探测器轴按 offset 与 real 对齐，返回可直接比对的子块。

    skimage radon(362×362) 给 512 个探测器 bin；LoDoPaB 给 513 个。
    多出来的那一个 bin 造成半像素级的错位，必须允许平移后再比。
    """
    if off > 0:
        return syn[:syn.shape[0] - off], real[off:]
    if off < 0:
        return syn[-off:], real[:real.shape[0] + off]
    return syn, real


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    with h5py.File(os.path.join(LODO, "ground_truth_test_000.hdf5"), "r") as f:
        gt = f["data"][:4].astype(np.float32)
    with h5py.File(os.path.join(LODO, "observation_test_000.hdf5"), "r") as f:
        obs = f["data"][:4].astype(np.float32)

    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)
    print(f"gt {gt.shape}   obs {obs.shape}")
    print(f"探测器 bin：skimage radon -> ", end="")
    syn0 = radon(gt[0], theta=ang, circle=False)
    print(f"{syn0.shape[0]} 个；  LoDoPaB -> {obs.shape[2]} 个\n")

    syn = np.stack([radon(gt[i], theta=ang, circle=False) * PIX_M
                    for i in range(len(gt))])          # (4, 512, 1000)
    real = obs.T                                        # (4, 513, 1000)

    # ---- 主测试：radon(gt) vs obs，各种变换 ----
    print(f"  {'变换':<34}{'corr':>10}")
    print("  " + "-" * 46)

    # 基准：需要把 512 vs 513 对齐
    best = (None, -2)
    for off in range(-3, 4):
        try:
            s, r = [], []
            for i in range(len(gt)):
                ss, rr = align_rows(syn[i], real[i], off)
                s.append(ss); r.append(rr)
            c = corr(np.concatenate(s), np.concatenate(r))
        except Exception:  # noqa: BLE001
            continue
        tag = f"radon(gt) vs obs, det offset={off:+d}"
        print(f"  {tag:<34}{c:>10.4f}")
        if c > best[1]:
            best = (off, c)

    off = best[0]
    print(f"\n  最佳探测器偏移 off = {off}（corr={best[1]:.4f}）\n")

    def cmp(tag, f):
        s, r = [], []
        for i in range(len(gt)):
            a, b = f(syn[i], real[i])
            s.append(a); r.append(b)
        c = corr(np.concatenate(s), np.concatenate(r))
        print(f"  {tag:<34}{c:>10.4f}")
        return c

    print(f"  {'变体':<34}{'corr':>10}")
    print("  " + "-" * 46)
    base = cmp("基准", lambda a, b: align_rows(a, b, off))
    cmp("obs 探测器轴翻转", lambda a, b: (align_rows(a, b, off)[0],
                                      align_rows(a, b, off)[1][::-1]))
    cmp("obs 角度轴翻转", lambda a, b: (align_rows(a, b, off)[0],
                                      align_rows(a, b, off)[1][:, ::-1]))
    cmp("radon(gt) 探测器轴翻转", lambda a, b: (align_rows(a, b, off)[0][::-1],
                                          align_rows(a, b, off)[1]))
    for k in (125, 250, 500):
        cmp(f"obs 角度轴整体平移 {k}", lambda a, b, k=k: np.roll(
            align_rows(a, b, off)[0], 0, axis=0)[:, :], ) if False else None
        c = 0.0
        s, r = [], []
        for i in range(len(gt)):
            a, b = align_rows(syn[i], real[i], off)
            s.append(a); r.append(np.roll(b, k, axis=1))
        c = corr(np.concatenate(s), np.concatenate(r))
        print(f"  {'obs 角度轴 roll %d' % k:<34}{c:>10.4f}")

    print("\n【判读】")
    print("  · 若某个变体 corr 明显高（>0.9），那就是约定差异所在，")
    print("    按它 FBP 即可复原出低剂量图")
    print("  · 若所有变体都接近 0，说明差异不在简单变换上——")
    print("    可能是探测器中心偏移（需要先平移再比），或角度不是均匀的")


if __name__ == "__main__":
    main()
