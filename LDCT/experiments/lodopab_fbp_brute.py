"""
最后一次穷举：sinogram 方向(2×2) × 重建结果的对称变换(8) = 32 种组合。

动机
------------------------------------------------------------------
`lodopab_fbp_null.py` 用 null 对照证明：先前"逐角度剖面相关 0.92–0.97"
**是假阳性**（不同角度互比也有 0.80–0.89）。胸部投影是宽钟形曲线，
近似对称，镜像了也看不出来。

故必须**直接对重建结果做全对称群穷举**，而不是继续拿弱指标猜。
若 32 种里有一种能把 corr 拉到 0.9 以上，方向问题即解决；
若全都不行，则重建本身是坏的，FBP 路径不通。

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_brute.py
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
from skimage.transform import iradon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LODO = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "outputs", "fbp_debug")


def corr(a, b):
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def symmetries(x):
    """二面体群 D4 的 8 个元素。"""
    out = {}
    for k in range(4):
        r = np.rot90(x, k)
        out[f"rot{k*90}"] = r
        out[f"rot{k*90}+flip"] = np.fliplr(r)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(OUT, exist_ok=True)

    with h5py.File(os.path.join(LODO, "ground_truth_test_000.hdf5"), "r") as f:
        gt = f["data"][0].astype(np.float64)
    with h5py.File(os.path.join(LODO, "observation_test_000.hdf5"), "r") as f:
        obs = f["data"][0].astype(np.float64)

    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)
    print(f"gt {gt.shape}  obs {obs.shape}\n")

    results = []
    for transposed in (True, False):
        for flip_det in (False, True):
            for flip_ang in (False, True):
                s = obs.T if transposed else obs
                if flip_det:
                    s = s[::-1]
                if flip_ang:
                    s = s[:, ::-1]
                if s.shape[0] not in (512, 513):
                    continue
                try:
                    rec = iradon(s, theta=ang, output_size=362,
                                 filter_name="ramp", circle=False)
                except Exception:  # noqa: BLE001
                    continue
                sino_tag = (f"{'T' if transposed else 'noT'}"
                            f"{'+fDet' if flip_det else ''}"
                            f"{'+fAng' if flip_ang else ''}")
                for sym_name, r in symmetries(rec).items():
                    results.append((corr(r, gt), sino_tag, sym_name, r))

    results.sort(key=lambda t: -t[0])
    print(f"  {'corr':>9}  {'sinogram':<18}{'输出变换'}")
    print("  " + "-" * 46)
    for c, st, sn, _ in results[:10]:
        print(f"  {c:>9.4f}  {st:<18}{sn}")
    print("  ...")
    print(f"  {results[-1][0]:>9.4f}  {results[-1][1]:<18}{results[-1][2]}"
          f"   <- 最差")

    best_c, best_st, best_sn, best_r = results[0]
    print(f"\n  最佳 corr = {best_c:.4f}  ({best_st} / {best_sn})")

    if best_c > 0.5:
        from PIL import Image
        lo, hi = best_r.min(), best_r.max()
        n = (best_r - lo) / (hi - lo) if hi > lo else np.zeros_like(best_r)
        Image.fromarray((n * 255).astype(np.uint8)).save(
            os.path.join(OUT, "50_best_brute.png"))
        print(f"  已存 {os.path.join(OUT, '50_best_brute.png')}")

    print("\n【判读】")
    print("  · corr > 0.9 => 找到了正确的方向组合，FBP 路径通")
    print("  · 全都 < 0.3 => 重建本身是坏的。那么问题不在方向，")
    print("    而在 sinogram 的**几何约定**（角度范围是 0-180 还是 0-360、")
    print("    探测器坐标定义），需要按 dival 的 ray_trafo 定义重写反投影。")


if __name__ == "__main__":
    main()
