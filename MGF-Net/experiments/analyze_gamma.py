"""
γ（梯度损失权重）扫描的自动判读。

要回答的问题
------------------------------------------------------------------
锐度与保真度之间是**此消彼长**（同一自由度的两端），
还是存在**支配点**（某个 γ 的锐度指标与保真度指标同时优于另一个 γ）？

- 若存在支配点 → 有免费午餐，选该工作点
- 若严格单调   → 模型表达能力不足，调参无用，**需改架构**

指标分类
------------------------------------------------------------------
无参考型（只看输出自身）：EN / SD / SF / Qabf
参考型  （衡量与源图关系）：MI / CC / PSNR / SSIM / VIF / SCD

用法
------------------------------------------------------------------
    python experiments/analyze_gamma.py
    python experiments/analyze_gamma.py --include-baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NO_REF = ["EN", "SD", "SF", "Qabf"]
REF = ["MI", "CC", "PSNR", "SSIM", "VIF", "SCD"]
ALL = NO_REF + REF


def load(label_prefix="gamma_"):
    by_gamma = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "results.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        c = os.path.join(os.path.dirname(p), "config.json")
        if not os.path.exists(c):
            continue
        cfg = json.load(open(c, encoding="utf-8"))
        lb = cfg.get("label", "")
        if not lb.startswith(label_prefix):
            continue
        g = float(lb.replace(label_prefix, ""))
        by_gamma[g].append(r["test"]["mean"])
    return by_gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-baseline", action="store_true")
    args = ap.parse_args()

    by_gamma = load()
    if not by_gamma:
        print("没有找到 gamma_* 结果")
        return

    gammas = sorted(by_gamma)
    print("=" * 104)
    print(f"γ 扫描结果（{len(gammas)} 个取值，每值 {len(by_gamma[gammas[0]])} 个种子）")
    print("=" * 104)
    print(f"  {'γ':<6}" + "".join(f"{k:>9}" for k in ALL) + f"{'n':>4}")
    print("  " + "-" * 100)
    means = {}
    for g in gammas:
        rows = by_gamma[g]
        m = {k: float(np.mean([r[k] for r in rows])) for k in ALL}
        s = {k: float(np.std([r[k] for r in rows])) for k in ALL}
        means[g] = (m, s)
        print(f"  {g:<6.0f}" + "".join(f"{m[k]:>9.4f}" for k in ALL) + f"{len(rows):>4}")

    if args.include_baseline:
        bp = os.path.join(HERE, "baselines_test.json")
        if os.path.exists(bp):
            bl = json.load(open(bp, encoding="utf-8"))["Average"]["mean"]
            print(f"  {'Avg':<6}" + "".join(f"{bl[k]:>9.4f}" for k in ALL) + "  ——")
            print("  ↑ Average = 零参数像素平均，作为对照下界")

    # ---------------- 单调性与支配点判定 ----------------
    print()
    print("=" * 104)
    print("判读")
    print("=" * 104)

    # 归一化到各指标自己的量程，便于跨指标比较"改进方向"
    def norm(k, v):
        vals = [means[g][0][k] for g in gammas]
        lo, hi = min(vals), max(vals)
        return (v - lo) / (hi - lo) if hi > lo else 0.5

    print("\n各指标随 γ 的趋势（+1 = 该指标在本次扫描中的最优端）：")
    print(f"  {'指标':<7}{'类别':<9}{'最优γ':>8}{'最差γ':>8}   单调性")
    print("  " + "-" * 60)
    mono = {}
    for k in ALL:
        vals = [means[g][0][k] for g in gammas]
        best_g = gammas[int(np.argmax(vals))]
        worst_g = gammas[int(np.argmin(vals))]
        d = np.diff(vals)
        is_mono = bool(np.all(d >= -1e-9) or np.all(d <= 1e-9))
        mono[k] = is_mono
        cat = "无参考" if k in NO_REF else "参考"
        print(f"  {k:<7}{cat:<9}{best_g:>8.0f}{worst_g:>8.0f}   "
              f"{'单调' if is_mono else '非单调'}")

    print()
    n_mono = sum(mono.values())
    print(f"  单调指标数: {n_mono}/{len(ALL)}")

    # 支配点：是否存在 (g1, g2) 使 g1 在无参考与参考两类上都 >= g2
    print()
    dominated = []
    for g1 in gammas:
        for g2 in gammas:
            if g1 == g2:
                continue
            ge_no = all(means[g1][0][k] >= means[g2][0][k] - 1e-9 for k in NO_REF)
            ge_ref = all(means[g1][0][k] >= means[g2][0][k] - 1e-9 for k in REF)
            if ge_no and ge_ref and (ge_no or ge_ref):
                if any(means[g1][0][k] > means[g2][0][k] + 1e-9 for k in ALL):
                    dominated.append((g1, g2))

    if dominated:
        print("  ⚠️ 发现**支配关系**（前者在所有指标上不劣于后者，且至少一项更优）：")
        for a, b in dominated:
            print(f"     γ={a:.0f} 支配 γ={b:.0f}")
        print("     → 存在免费午餐。可在不被支配的 γ 中选锐度更优者。")
    else:
        print("  ✅ 未发现支配关系 —— 锐度与保真度**严格此消彼长**。")
        print("     含义：模型只能在「像平均」与「锐但失真」之间滑动，")
        print("     这是**表达能力不足**的表现，调 γ 无法解决，需改架构。")

    # 与平凡平均比
    bp = os.path.join(HERE, "baselines_test.json")
    if os.path.exists(bp):
        bl = json.load(open(bp, encoding="utf-8"))["Average"]["mean"]
        print()
        print("  相对平凡平均（Average）的优势（正=更好）：")
        print(f"  {'γ':<6}" + "".join(f"{k:>9}" for k in ALL))
        print("  " + "-" * 76)
        for g in gammas:
            print(f"  {g:<6.0f}" + "".join(
                f"{means[g][0][k] - bl[k]:>+9.4f}" for k in ALL))

    print()
    print("⚠️  2 个种子、2 个测试病例。差异普遍在噪声量级，")
    print("    本节仅用于判断**趋势方向**，不得用于声称显著性。")


if __name__ == "__main__":
    main()
