"""
定位 LoDoPaB FBP 的旋转中心偏移。

已确认的事实
------------------------------------------------------------------
  · corr( radon_skimage(gt)×0.000718 , obs.T ) = **0.970**
    => 真实 sinogram 与 skimage 的约定一致（含尺度）
  · iradon(radon_skimage(gt)) 与 gt 的相关度 = **0.998**  => iradon 用法对
  · 但 iradon(obs.T) 与 gt 的相关度只有 **0.03**
  两者内容 97% 相同却一个成一个不成 —— 排除"约定不同"，
  只剩**探测器 bin 数差 1（512 vs 513）造成的旋转中心偏移**。

半像素的旋转中心偏移在 FBP 里会产生"音叉"状拖尾（调试图里已肉眼可见），
并让相关度塌掉。

本脚本遍历裁剪/平移方案，找出使重建与 gt 相关度最高的中心。

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_center.py
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
OUT = os.path.join(ROOT, "outputs", "fbp_debug")
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
    os.makedirs(OUT, exist_ok=True)

    with h5py.File(os.path.join(LODO, "ground_truth_test_000.hdf5"), "r") as f:
        gt = f["data"][:3].astype(np.float64)
    with h5py.File(os.path.join(LODO, "observation_test_000.hdf5"), "r") as f:
        obs = f["data"][:3].astype(np.float64)

    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)

    # ---- 基准：合成路径（已知能成）----
    print("【基准】合成路径")
    rec = iradon(radon(gt[0], theta=ang, circle=False), theta=ang,
                 output_size=362, filter_name="ramp", circle=False)
    print(f"  iradon(radon(gt)) vs gt : corr = {corr(rec, gt[0]):.4f}"
          f"   （应为 ~0.998）\n")

    print("【裁剪方案】把 513 bin 裁成不同长度/位置，看重建相关度")
    print(f"  {'方案':<38}{'corr':>10}")
    print("  " + "-" * 50)

    best = (None, -2, None)
    s = obs[0].T                       # (513, 1000)

    schemes = []
    # 裁掉首位不同数量的 bin
    for drop_start in (0, 1):
        for length in (512, 511, 510):
            schemes.append((f"drop_start={drop_start}, len={length}",
                            s[drop_start:drop_start + length]))
    # 不裁，直接给 iradon
    schemes.append(("原样 513", s))

    for name, sino in schemes:
        try:
            r = iradon(sino, theta=ang, output_size=362,
                       filter_name="ramp", circle=False)
            c = corr(r, gt[0])
            print(f"  {name:<38}{c:>10.4f}")
            if c > best[1]:
                best = (name, c, r)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<38}  失败 {type(e).__name__}")

    # 亚像素平移：用线性插值把 sinogram 整体平移 frac 个 bin
    print(f"\n  {'亚像素平移（裁 512 后）':<38}{'corr':>10}")
    print("  " + "-" * 50)
    from scipy.ndimage import shift as ndshift
    for frac in (-0.5, -0.25, 0.0, 0.25, 0.5):
        try:
            moved = ndshift(s[:512], (frac, 0.0), order=1, mode="nearest")
            r = iradon(moved, theta=ang, output_size=362,
                       filter_name="ramp", circle=False)
            c = corr(r, gt[0])
            print(f"  {'shift=%+.2f bin' % frac:<38}{c:>10.4f}")
            if c > best[1]:
                best = (f"shift {frac:+.2f}", c, r)
        except Exception as e:  # noqa: BLE001
            print(f"  {'shift=%+.2f' % frac:<38}  失败 {type(e).__name__}")

    print("  " + "-" * 50)
    print(f"\n  最佳：{best[0]}   corr = {best[1]:.4f}")

    if best[2] is not None:
        from PIL import Image
        r = best[2]
        lo, hi = r.min(), r.max()
        n = (r - lo) / (hi - lo) if hi > lo else np.zeros_like(r)
        Image.fromarray((n * 255).astype(np.uint8)).save(
            os.path.join(OUT, "40_best_center.png"))
        print(f"  已存 {os.path.join(OUT, '40_best_center.png')}")

    print("\n【判读】")
    print("  · 若某方案 corr > 0.9 => 中心问题解决，FBP 路径打通")
    print("  · 若全都很低 => skimage 的 resize 逻辑（output_size < 探测器数时\n"
          "    会对 sinogram 做 order=0 重采样）才是元凶，需绕开它")


if __name__ == "__main__":
    main()
