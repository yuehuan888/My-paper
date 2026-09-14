"""
LoDoPaB FBP 方向穷举：把（转置 / 探测器轴翻转 / 角度轴翻转）的 8 种组合全试一遍。

依据（lodopab_fbp_debug.py 的图，肉眼可见）：
  · 真实 obs 与 skimage radon(gt) 的 sinogram 形态一致 —— 转置方向对
  · 但 iradon(obs) 重建出的图**是 gt 的镜像/180° 旋转**，并带严重拖尾
    => 典型的"探测器轴方向反了"
看图像比看指标快得多，但确认要靠穷举打分。

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_orient.py
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


def corr(a, b, w=500):
    """居中裁到同样宽度再比。

    skimage radon(362×362) 给 512 个探测器 bin，LoDoPaB 给 513 个，
    直接相减会广播失败。两者中心对齐后取中间 w 个 bin。
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape[0] != b.shape[0]:
        ca, cb = a.shape[0] // 2, b.shape[0] // 2
        a = a[ca - w // 2: ca + w // 2]
        b = b[cb - w // 2: cb + w // 2]
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def fit_and_score(rec, gt):
    """仿射拟合后的相关度（尺度与偏移都不该影响判定）。"""
    a = np.stack([gt.ravel(), np.ones(gt.size)], 1)
    sol, *_ = np.linalg.lstsq(a, rec.ravel(), rcond=None)
    pred = (a @ sol).reshape(gt.shape)
    return corr(pred, rec), sol


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

    # 先确认合成 sinogram 与真实 sinogram 的对应关系
    ss = radon(gt[0], theta=ang, circle=False) * PIX_M
    print(f"合成 sinogram {ss.shape}  真实 obs[0] {obs[0].shape}")
    d_flip = corr(ss, obs[0].T)
    print(f"  corr(radon(gt)*PIX_M, obs.T) = {d_flip:+.4f}"
          f"   （轴序已对齐：探测器 × 角度）")
    print("  注：obs 原形 (1000,513)=（角度,探测器），radon 给 (512,1000)=（探测器,角度），"
          "故必须转置后才能比。")

    print(f"\n  {'组合':<40}{'corr(拟合后)':>14}{'a':>12}")
    print("  " + "-" * 68)

    best = (None, -2, None)
    for transp in (True, False):
        for flip_det in (False, True):
            for flip_ang in (False, True):
                recs = []
                for i in range(len(gt)):
                    s = obs[i].T if transp else obs[i]
                    if flip_det:
                        s = s[::-1]
                    if flip_ang:
                        s = s[:, ::-1]
                    if s.shape[0] != 513:
                        continue
                    recs.append(iradon(s, theta=ang, output_size=362,
                                       filter_name="ramp", circle=False))
                if not recs:
                    continue
                c_sum, a_sum = 0.0, 0.0
                for r, g in zip(recs, gt):
                    c, sol = fit_and_score(r, g)
                    c_sum += c
                    a_sum += sol[0]
                c_mean = c_sum / len(recs)
                name = (f"{'T' if transp else 'noT'}"
                        f"{'/flipDet' if flip_det else ''}"
                        f"{'/flipAng' if flip_ang else ''}")
                print(f"  {name:<40}{c_mean:>14.4f}{a_sum/len(recs):>12.6f}")
                if c_mean > best[1]:
                    best = (name, c_mean, recs[0])

    print("  " + "-" * 68)
    print(f"\n  最佳：{best[0]}   corr = {best[1]:.4f}")

    if best[2] is not None:
        from PIL import Image
        r = best[2]
        lo, hi = r.min(), r.max()
        n = (r - lo) / (hi - lo) if hi > lo else np.zeros_like(r)
        Image.fromarray((n * 255).astype(np.uint8)).save(
            os.path.join(OUT, "30_best_orient.png"))
        print(f"  已存 {os.path.join(OUT, '30_best_orient.png')}")

    print("\n【判读】")
    print("  · corr > 0.9 => 方向组合找对了，FBP 路径可行")
    print("  · 若全都 < 0.3 => 差异不在方向，而在探测器中心偏移或角度范围")


if __name__ == "__main__":
    main()
