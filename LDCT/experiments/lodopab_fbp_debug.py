"""
LoDoPaB FBP 调试：把图存出来直接看，而不是靠指标猜。

`lodopab_fbp_probe.py` 的结果：FBP 输出值域仅 ~5e-4、拟合后 PSNR -0.55 dB。
输出**不是全零**、且线性拟合斜率稳定，说明结构可能在其中，只是量级/朝向不对。
指标看不出"错在哪"，图像可以。

本脚本做三件事：
  1. 合成往返测试：拿 LoDoPaB 的 gt 图 → skimage.radon 正投影 → iradon 反投影
     若往返能复原，说明 iradon 用法本身没问题，锅在真实 sinogram 的约定上
  2. 真实数据 FBP：多种参数变体（角度范围/朝向/是否转置），各自归一化后存图
  3. 全部存成 PNG，人工比对

用法
------------------------------------------------------------------
    python experiments/lodopab_fbp_debug.py
然后看 outputs/fbp_debug/ 下的图。
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
from skimage.transform import iradon, radon, resize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LODO = os.path.join(ROOT, "data", "extracted")
OUT = os.path.join(ROOT, "outputs", "fbp_debug")


def norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def save(name, a):
    from PIL import Image
    Image.fromarray((norm01(a) * 255).astype(np.uint8)).save(
        os.path.join(OUT, name))
    print(f"    存 {name}   值域[{a.min():+.6f},{a.max():+.6f}] "
          f"std={a.std():.6f}")


def corr(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    os.makedirs(OUT, exist_ok=True)

    with h5py.File(os.path.join(LODO, "ground_truth_test_000.hdf5"), "r") as f:
        gt = f["data"][0].astype(np.float32)
    with h5py.File(os.path.join(LODO, "observation_test_000.hdf5"), "r") as f:
        obs = f["data"][0].astype(np.float32)

    print(f"gt {gt.shape} [{gt.min():.4f},{gt.max():.4f}]   "
          f"obs {obs.shape} [{obs.min():.4f},{obs.max():.4f}]\n")

    save("00_ground_truth.png", gt)
    save("01_sinogram.png", obs)

    # ---------------------------------------------------------- 1 合成往返
    print("\n【1】合成往返：gt --radon--> sino --iradon--> ?")
    ang = np.linspace(0.0, 180.0, 1000, endpoint=False)
    sino_syn = radon(gt, theta=ang, circle=False)
    print(f"    合成 sinogram {sino_syn.shape} 值域 "
          f"[{sino_syn.min():.4f},{sino_syn.max():.4f}]")
    save("02_syn_sinogram.png", sino_syn)
    rec_syn = iradon(sino_syn, theta=ang, output_size=gt.shape[0],
                     filter_name="ramp", circle=False)
    print(f"    往返重建相关度 corr = {corr(rec_syn, gt):.6f}")
    save("03_syn_roundtrip.png", rec_syn)

    # ---------------------------------------------------------- 2 真实数据
    print("\n【2】真实 observation 的各种变体")
    variants = {
        "10_real_T": obs.T,
        "11_real_noT": obs,
    }
    for name, sino in variants.items():
        try:
            rec = iradon(sino, theta=ang, output_size=362,
                         filter_name="ramp", circle=False)
            c = corr(rec, gt)
            print(f"  {name}: shape_in={sino.shape} corr(rec,gt)={c:.4f}")
            save(f"{name}.png", rec)
        except Exception as e:  # noqa: BLE001
            print(f"  {name}: 失败 {type(e).__name__}: {e}")

    # ---------------------------------------------------------- 3 角度/滤波器
    print("\n【3】滤波器与角度变体（均用 obs.T）")
    for fname in ["ramp", "shepp-logan", "hann"]:
        rec = iradon(obs.T, theta=ang, output_size=362,
                     filter_name=fname, circle=False)
        print(f"    filter={fname:<12} corr={corr(rec, gt):.4f}")
        save(f"20_filt_{fname}.png", rec)

    rec = iradon(obs.T, theta=ang, output_size=513, filter_name="ramp",
                 circle=False)
    print(f"    output_size=513         corr={corr(rec, gt):.4f}")
    save("21_outsize513.png", rec)
    save("22_outsize513_cropped.png", rec[76:76 + 362, 76:76 + 362])

    print("\n【判读】")
    print("  · 若 03_syn_roundtrip 能看出 gt 的形状 => iradon 用法没错")
    print("  · 若 10_real_T / 11_real_noT 里有一个能看出人体轮廓 => 转置方向对了")
    print("  · 若全都看不出轮廓 => 角度范围(0..180 vs 0..360)或其他约定不对")


if __name__ == "__main__":
    main()
