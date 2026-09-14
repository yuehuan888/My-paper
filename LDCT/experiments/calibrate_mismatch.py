"""
标定干预实验的注入强度 ε：使 PR 臂的闭环误差达到无约束臂的实测水平。

背景
------------------------------------------------------------------
论文 §5.3 报告：无约束臂训练后的闭环相对 L2 误差为 **1.25e-01（S1，5 种子均值）**，
而两个可逆臂停在 1.6e-07。机制主张是"**这个**误差导致了性能退化"。

要把它从归因变成因果，就得做干预：**给 PR 臂注入同样量级的误差，看性能是否退化**。
注入方式：合成步用 (1−ε)·U，分解仍用 U（见 models/lifting_dwt.py 的 synth_mismatch）。

本脚本找出使闭环误差 ≈ 1.25e-01 的 ε。

测法必须与论文一致
------------------------------------------------------------------
`verify_roundtrip.py` 用的是**固定种子的随机 256×256 图**。为了让干预臂与无约束臂
的数字可比，这里沿用同一个探针（同尺寸、同种子），而不是换成真实 CT 切片——
**换探针会让"12.5%"这个数失去可比性**。

误差随 ε 的尺度
------------------------------------------------------------------
一阶上：xe' = xe + ε·U(d)，xo' = xo + ε·P(U(d))，即误差 ∝ ε。
但 U 在训练中会漂移，故**初始化时标定的 ε，训练后测得的值会不同**。
本脚本报初值，训练后再用同一探针复测（与论文对 unconstrained 臂的处理一致）。

用法
------------------------------------------------------------------
    python experiments/calibrate_mismatch.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from models.denoiser import PRWaveletDenoiser     # noqa: E402

TARGET = 1.253e-01          # 无约束臂 S1 五种子均值（论文 §5.3）
TARGET_LO, TARGET_HI = 0.11, 0.15   # S1 逐种子范围


def probe(model, size=256, seed=7):
    """与 verify_roundtrip.py 完全相同的探针，保证数字可比。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(1, 1, size, size, generator=g)
    me, rel = model.transform_roundtrip(x)
    return me, rel


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="*", default=None,
                    help="要试的 ε 列表；默认自动扫")
    args = ap.parse_args()

    eps_list = args.eps if args.eps else [
        0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print(f"目标：闭环 rel_L2 ≈ {TARGET:.3e}"
          f"（无约束臂 S1 实测；逐种子范围 {TARGET_LO}–{TARGET_HI}）\n")
    print(f"  {'ε':>7}{'max|err|':>13}{'rel_L2':>13}   与目标比")
    print("  " + "-" * 52)

    rows = []
    for eps in eps_list:
        torch.manual_seed(0)
        m = PRWaveletDenoiser(levels=2, mid_ch=16, n_conv=2, wavelet="pr",
                              synth_mismatch=eps).eval()
        me, rel = probe(m)
        ratio = rel / TARGET
        mark = ""
        if TARGET_LO <= rel <= TARGET_HI:
            mark = "  ← 落在无约束臂的种子范围内"
        print(f"  {eps:>7.3f}{me:>13.3e}{rel:>13.3e}   {ratio:>6.3f}×{mark}")
        rows.append({"eps": eps, "max_abs": me, "rel_l2": rel, "ratio": ratio})

    # 线性插值给出建议值
    print("\n  " + "-" * 52)
    # 找跨越目标的两点做线性插值（rel 对 ε 近似线性）
    best = min(rows, key=lambda r: abs(r["rel_l2"] - TARGET))
    lo = max([r for r in rows if r["rel_l2"] <= TARGET] or [rows[0]],
             key=lambda r: r["rel_l2"])
    hi = min([r for r in rows if r["rel_l2"] >= TARGET] or [rows[-1]],
             key=lambda r: r["rel_l2"])
    sug = best["eps"]
    if hi["rel_l2"] > lo["rel_l2"]:
        # 线性插值
        f = (TARGET - lo["rel_l2"]) / (hi["rel_l2"] - lo["rel_l2"])
        sug = lo["eps"] + f * (hi["eps"] - lo["eps"])
        print(f"  线性插值建议 ε ≈ {sug:.4f}"
              f"（在 ε={lo['eps']} 与 ε={hi['eps']} 之间）")
    else:
        print(f"  建议 ε = {sug}")

    print(f"\n  注：这是**初始化时**的值。训练中 U 会漂移，实际误差需训练后复测。")

    dst = os.path.join(HERE, "calibrate_mismatch.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"target": TARGET, "target_range": [TARGET_LO, TARGET_HI],
                   "suggested_eps": sug, "scan": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"  已写入 {dst}")


if __name__ == "__main__":
    main()
