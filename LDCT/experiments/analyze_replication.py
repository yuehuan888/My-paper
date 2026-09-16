"""复现研究分析：新数据集上的三臂对照 + bound 剂量-响应。

背景
================================================================
论文的原始结果基于一次数据获取（435 片测试集）。后因官方 Box 不可达、
Kaggle 镜像下架，数据被**重新获取**（421 片，缺 14 张）。本脚本分析在
这批新数据上重跑的全部实验，回答两个问题：

  1. 论文的三个核心结论能否在新数据上复现？（三臂对照，各 n=10 种子）
  2. 贡献 3 的 bound 界定，是一条什么样的曲线？（5 档 × 3 种子）

输出
================================================================
  - analysis_replication.json  —— 全部数字
  - paper/figures/fig10..fig13 —— 复现图、bound 曲线、效应量对比、种子收敛
"""

from __future__ import annotations

import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "analysis_replication.json")

# 论文原始值（S1，435 片测试集，n=5）—— 用于对比复现
PAPER = {
    "pr": (30.8314, 0.0815),
    "unconstrained": (30.5341, 0.0305),
    "fixed": (30.5591, 0.0256),
    "cmp": {
        "pr-unconstrained": (+0.2973, 0.0005, +4.67),
        "pr-fixed": (+0.2723, 0.0004, +4.78),
        "unconstrained-fixed": (-0.0251, 0.1131, -0.90),
    },
}
# bound 消融的档位（0.5 就是默认，tag 用 new_pr_s*）
BOUNDS = [(0.5, "new_pr_s%d"), (1.0, "new_bnd10_s%d"), (2.0, "new_bnd20_s%d"),
          (4.0, "new_bnd40_s%d"), (8.0, "new_bnd80_s%d")]


def psnr(tag):
    p = os.path.join(RUNS, tag, "results.json")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)["test_mean"]["PSNR"]


def ssim(tag):
    p = os.path.join(RUNS, tag, "results.json")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)["test_mean"]["SSIM"]


def seed_series(arm, n_seeds=10, prefix="new_"):
    v = []
    for s in range(n_seeds):
        x = psnr(f"{prefix}{arm}_s{s}")
        if x is not None:
            v.append(x)
    return np.array(v)


def paired(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 2 or len(a) != len(b):
        return None
    d = a - b
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return {
        "delta": float(d.mean()), "t": float(t), "p": float(p), "n": n,
        "ci": [float(d.mean() - crit * se), float(d.mean() + crit * se)],
        "dz": float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan"),
    }


def mdes(n, alpha=0.05, power=0.80):
    from scipy import stats
    if n < 2:
        return float("nan")
    return float((stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(power, n - 1))
                 / np.sqrt(n))


def main():
    R = {"arms": {}, "comparisons": {}, "bound": {}, "paper": PAPER}

    print("=" * 74)
    print("复现研究：新数据集（421 片测试集）")
    print("=" * 74)

    # ---------------- 三臂 ----------------
    print("\n[1] 三臂对照（n=10 种子）")
    arm_data = {}
    for arm in ("pr", "unconstrained", "fixed"):
        v = seed_series(arm, 10)
        arm_data[arm] = v
        if len(v) == 0:
            continue
        s = ssim(f"new_{arm}_s0")
        R["arms"][arm] = {
            "n_seeds": len(v), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "seed_means": [float(x) for x in v],
            "ssim_s0": s,
            "paper_mean": PAPER[arm][0], "paper_sd": PAPER[arm][1],
        }
        pn = PAPER[arm]
        print(f"  {arm:<14} 新 {v.mean():.4f} ± {v.std(ddof=1):.4f} (n={len(v)})   "
              f"论文 {pn[0]:.4f} ± {pn[1]:.4f}   差 {v.mean()-pn[0]:+.4f}")

    # ---------------- 配对比较 ----------------
    print("\n[2] 配对比较（n=10）")
    for a, b in (("pr", "unconstrained"), ("pr", "fixed"), ("unconstrained", "fixed")):
        if len(arm_data[a]) == 0 or len(arm_data[a]) != len(arm_data[b]):
            continue
        c = paired(arm_data[a], arm_data[b])
        if c is None:
            continue
        key = f"{a}-{b}"
        pp = PAPER["cmp"].get(key)
        c["paper"] = {"delta": pp[0], "p": pp[1], "dz": pp[2]} if pp else None
        c["mdes_dB"] = mdes(c["n"]) * float(arm_data[a].std(ddof=1))
        c["delta_over_mdes"] = abs(c["delta"]) / c["mdes_dB"] if c["mdes_dB"] > 0 else float("nan")
        R["comparisons"][key] = c
        sig = "***" if c["p"] < 0.001 else ("**" if c["p"] < 0.01 else
                                           ("*" if c["p"] < 0.05 else "n.s."))
        print(f"  {key:<24} Δ={c['delta']:+.4f}  p={c['p']:.4f} {sig:<4} "
              f"dz={c['dz']:+.2f}   |Δ|/MDES={c['delta_over_mdes']:.2f}×"
              + (f"   论文 Δ={pp[0]:+.4f}" if pp else ""))

    # ---------------- bound 曲线 ----------------
    print("\n[3] bound 剂量-响应")
    for b, pat in BOUNDS:
        v = [psnr(pat % s) for s in range(3)]
        v = [x for x in v if x is not None]
        if not v:
            continue
        R["bound"][str(b)] = {
            "n_seeds": len(v), "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "seeds": [float(x) for x in v],
        }
        print(f"  bound={b:<4} {np.mean(v):.4f} ± "
              f"{(np.std(v, ddof=1) if len(v)>1 else 0):.4f}  (n={len(v)})")

    if R["bound"]:
        best = max(R["bound"].items(), key=lambda kv: kv[1]["mean"])
        dflt = R["bound"].get("0.5", {}).get("mean")
        print(f"\n  峰值 bound={best[0]}  {best[1]['mean']:.4f}")
        if dflt:
            print(f"  默认 0.5 = {dflt:.4f}，比峰值低 {best[1]['mean']-dflt:+.4f} dB")
        R["bound_best"] = {"bound": best[0], "mean": best[1]["mean"],
                           "gain_over_default": (best[1]["mean"] - dflt) if dflt else None}

    # ---------------- 种子收敛 ----------------
    print("\n[4] 均值随种子数的收敛（pr 臂）")
    v = arm_data.get("pr", np.array([]))
    R["convergence"] = {}
    if len(v) > 1:
        for n in range(2, len(v) + 1):
            R["convergence"][str(n)] = {
                "mean": float(v[:n].mean()),
                "halfwidth": float(1.96 * v[:n].std(ddof=1) / np.sqrt(n)),
            }
        print(f"  n=2 均值 {v[:2].mean():.4f}  →  n={len(v)} 均值 {v.mean():.4f}")

    with io.open(OUT, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    print(f"\n已写入 {OUT}")
    return R


if __name__ == "__main__":
    main()
