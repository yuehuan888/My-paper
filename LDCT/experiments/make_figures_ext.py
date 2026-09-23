"""扩展图表：只用**已有产物**产出论文还没用过的 5 张图（零算力）。

与 `make_figures.py` 的分工
================================================================
`make_figures.py` 产出 fig1~fig4（三臂柱状、机制、视觉、Pareto）。
本文件产出 fig5~fig9，全部基于 `analysis_extended.py` 的结果。

配色（**经 dataviz 校验器实测通过，非凭感觉**）
================================================================
  node scripts/validate_palette.js "#2E5EAA,#D1495B,#7B52AB" --mode light
    [PASS] 亮度带 / 色度下限 / CVD 分离(ΔE 14.3 protan) / 常视觉下限(ΔE 20.0) / 对比度
    → ALL CHECKS PASS

原先打算给 `fixed` 用中性灰 #6C757D —— **实测未通过**：
  色度 0.016（"读起来是灰的"，低于 0.1 下限）+ 与红色的 protan ΔE 仅 5.9。
故改用紫色 #7B52AB。灰色只保留给**参考线**（参考线不是分类系列，不受此约束）。

制图规范（同样来自该 skill）
================================================================
  - **绝不双 y 轴**（两个不同量纲 → 拆成两张图或归一化）
  - ≥2 个系列必有图例；文字用墨色而非系列色
  - 细线（linewidth 2）、markersize ≥ 8、网格弱化
  - 直接标注有选择地加，不是每个点都标数字
"""

from __future__ import annotations

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "..", "paper", "figures")
RJSON = os.path.join(HERE, "analysis_extended.json")

# ---- 经校验器通过的配色 -------------------------------------------------
C = {"pr": "#2E5EAA", "unconstrained": "#D1495B", "fixed": "#7B52AB"}
LAB = {"pr": "PR-LWT (ours)", "unconstrained": "Unconstrained", "fixed": "Fixed Haar"}
ORDER = ["pr", "unconstrained", "fixed"]

# 墨色（文字一律用这些，不用系列色）
INK = "#1a1a19"
INK2 = "#55534d"
MUTED = "#8a8880"
GRID = "#e3e2dd"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "figure.dpi": 300,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load():
    with io.open(RJSON, encoding="utf-8") as f:
        return json.load(f)


# ============================================================ fig5 分布
def fig5_distribution(R, out):
    """逐切片 PSNR 分布 —— 证明效应不是被少数切片带出来的。

    形式选择：要展示**分布**而非汇总量 → 半小提琴 + 抖动散点（raincloud）。
    比柱状图信息量大得多，且能同时看出重叠程度。
    """
    S = R["splits"]["S1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [2.1, 1]})

    # ---- (a) 逐切片 PSNR 分布 ----
    ax = axes[0]
    M, tags = [], []
    for arm in ORDER:
        p = os.path.join(HERE, "runs", "%s_pr_s0" % "w3")
        # 直接从 analysis_extended 的原始 runs 重新读，保证是逐切片真值
        ps = []
        for s in range(S["arms"][arm]["n_seeds"]):
            tag = S["arms"][arm]["tags"][s]
            with io.open(os.path.join(HERE, "runs", tag, "results.json"), encoding="utf-8") as f:
                d = json.load(f)
            ps.append([r["PSNR"] for r in d["test_per_sample"]])
        M.append(np.concatenate(ps))

    parts = ax.violinplot(M, positions=range(len(ORDER)), widths=0.72,
                          showextrema=False, showmedians=False)
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(C[ORDER[i]])
        b.set_alpha(0.30)
        b.set_edgecolor("none")

    rng = np.random.RandomState(0)
    for i, v in enumerate(M):
        y = v[rng.choice(len(v), 400, replace=False)]
        ax.scatter(i + rng.normal(0, 0.045, len(y)), y, s=1.6,
                   color=C[ORDER[i]], alpha=0.30, linewidths=0, zorder=2)
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([i - 0.20, i + 0.20], [med, med], color=INK, lw=2, zorder=4)
        ax.plot([i, i], [q1, q3], color=INK, lw=4, solid_capstyle="butt", zorder=3)
        ax.annotate("%.2f" % med, (i, med), xytext=(0, 0), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7.5, color=INK,
                    zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc=SURFACE,
                                        ec="none", alpha=0.85))

    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels([LAB[a] for a in ORDER], fontsize=8.5)
    ax.set_ylabel("Per-slice PSNR (dB)")
    ax.set_title("(a) Distribution over all 435 test slices (S1)", loc="left", pad=8)
    ax.grid(axis="x", visible=False)
    despine(ax)
    ylo, yhi = min(v.min() for v in M), max(v.max() for v in M)
    ax.set_ylim(ylo - 0.6, yhi + 1.4)

    # ---- (b) 种子间 vs 种子内 变异分解 ----
    ax = axes[1]
    x = np.arange(len(ORDER))
    within = [S["arms"][a]["within_seed_sd"] for a in ORDER]
    across = [S["arms"][a]["across_seed_sd"] for a in ORDER]
    h = 0.34
    ax.bar(x - h / 2, within, h, color=[C[a] for a in ORDER], alpha=0.9,
           edgecolor=SURFACE, linewidth=2, label="Within-seed (slice) SD")
    ax.bar(x + h / 2, across, h, color=[C[a] for a in ORDER], alpha=0.42,
           edgecolor=SURFACE, linewidth=2, hatch="///", label="Across-seed SD")
    for i in range(len(ORDER)):
        ax.annotate("%.2f" % within[i], (i - h / 2, within[i]), ha="center",
                    va="bottom", fontsize=7.5, color=INK)
        ax.annotate("%.3f" % across[i], (i + h / 2, across[i]), ha="center",
                    va="bottom", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["PR-LWT", "Unconstr.", "Fixed"], fontsize=8.5)
    ax.set_ylabel("SD of PSNR (dB)")
    ax.set_title("(b) Variance decomposition", loc="left", pad=8)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.set_ylim(0, max(within) * 1.28)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ======================================================== fig6 训练动力学
def fig6_convergence(R, out):
    """训练过程 —— 论文从未分析过 history.json。

    形式：两条**独立**的曲线图（训练 L1 / 验证 PSNR）。
    **不画双 y 轴** —— 两者量纲不同，双轴会制造虚假的交叉点。
    """
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    # ⚠️ 两个面板的**横轴不同**：训练每轮一个点，验证每 val_interval 轮一个点。
    #    必须各用各的 x 坐标，否则验证点会被压缩到时间轴前 1/5（实测踩过）。
    for ax, key, xkey, ylab, title in (
        (axes[0], "train_l1_curve_mean", "train_epochs", "Training L1 loss", "(a) Training loss"),
        (axes[1], "val_psnr_curve_mean", "val_epochs", "Validation PSNR (dB)", "(b) Validation PSNR"),
    ):
        for arm in ORDER:
            cs = R["splits"]["S1"]["convergence"][arm]
            m = np.array(cs[key])
            sd = np.array(cs["train_l1_curve_sd"] if "l1" in key
                          else cs["val_psnr_curve_sd"])
            ep = np.array(cs[xkey][:len(m)])
            ax.plot(ep, m, color=C[arm], lw=2, label=LAB[arm],
                    solid_capstyle="round",
                    marker="o" if "val" in key else None, ms=4 if "val" in key else None)
            ax.fill_between(ep, m - sd, m + sd, color=C[arm], alpha=0.15, linewidth=0)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", pad=8)
        despine(ax)
        ax.set_xlim(1, 30)

    # 标出"最后一轮仍是最优轮"这件事 —— 论文没提，但很重要
    axes[1].axvline(30, color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    axes[1].annotate("best epoch = 30\nfor all seeds\n(not converged)",
                     xy=(30, axes[1].get_ylim()[0]), xytext=(-8, 10),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=7, color=INK2, linespacing=1.35)
    axes[0].legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ========================================================== fig7 森林图
def fig7_forest(R, out):
    """效应量森林图 —— 所有配对比较 + 最小可检测效应带。

    形式选择：这是**效应量 + 不确定区间**的数据，森林图是标准形式。
    额外画上 MDES 边界：落在带内的比较等于"这个 n 根本测不出来"。
    """
    rows = []
    for split in ("S1", "S2"):
        for k, c in R["splits"][split]["comparisons"].items():
            mde = R["splits"][split]["power"]["n=%d" % c["n"]]["mdes_in_dB"]
            rows.append({"split": split, "cmp": k, **c, "mde": mde})

    # 按划分分组、组内按效应量排序
    rows.sort(key=lambda r: (r["split"], -abs(r["delta"])))
    fig, ax = plt.subplots(figsize=(7.0, 3.4))

    ypos, ylabels, seps = [], [], []
    y = 0
    for split in ("S1", "S2"):
        for r in [q for q in rows if q["split"] == split]:
            ypos.append(y)
            a, b = r["cmp"].split("-")
            ylabels.append("%s  −  %s" % (LAB[a].replace(" (ours)", ""), LAB[b].replace(" (ours)", "")))
            y += 1
        seps.append(y - 0.5)
        y += 0.9

    for i, r in enumerate(rows):
        yy = ypos[i]
        sig = r["p"] < 0.05
        col = C[r["cmp"].split("-")[0]] if sig else MUTED
        # MDES 带（该 n 下测不出来的区间）
        ax.barh(yy, 2 * r["mde"], left=-r["mde"], height=0.62,
                color=GRID, alpha=0.55, zorder=1, linewidth=0)
        ax.plot([r["ci"][0], r["ci"][1]], [yy, yy], color=col, lw=2.2,
                solid_capstyle="round", zorder=3)
        ax.scatter([r["delta"]], [yy], s=64, color=col, zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate("%+.3f  (p=%.3f)" % (r["delta"], r["p"]),
                    (r["ci"][1], yy), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=INK if sig else INK2)

    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("Δ PSNR (dB)   [positive = left arm better]")
    ax.set_title("Paired comparisons (unit = training seed), 95% CI",
                 loc="left", pad=8)
    ax.grid(axis="y", visible=False)
    despine(ax)
    ax.invert_yaxis()

    hi = max(r["ci"][1] for r in rows)
    lo = min(r["ci"][0] for r in rows)
    ax.set_xlim(lo - 0.10, hi + 0.30)

    # 分组标签：画在**每组上方**。轴已反转，故"上方" = 更小的 y。
    # ⚠️ 必须显式给首尾留白，否则第一组的标签会被裁到画布外（实测踩过）。
    y0 = 0
    for split in ("S1", "S2"):
        grp = [q for q in rows if q["split"] == split]
        ax.annotate("%s (n=%d seeds)" % (split, grp[0]["n"]),
                    (ax.get_xlim()[0] + 0.02, y0 - 0.80), fontsize=7.5,
                    color=INK2, va="center", zorder=6)
        y0 += len(grp) + 0.9
    ax.set_ylim(ypos[-1] + 0.70, ypos[0] - 1.30)

    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=2.2, marker="o", ms=6, ls="none",
               markeredgecolor=SURFACE, label="n.s. (p ≥ 0.05)"),
        Patch(facecolor=GRID, alpha=0.55, label="MDES band — below this n's detection floor"),
    ], fontsize=7.5, loc="lower right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ========================================================= fig8 功效曲线
def fig8_power(R, out):
    """功效分析 —— 正面回答"n=5 够不够"。

    论文原来的说法是"同一实验能分辨 +0.27dB，所以不是功效不足"。
    这张图把它变成定量的：n 与可检测效应的关系，以及观测效应落在哪。
    """
    S = R["splits"]["S1"]
    ns = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50])
    from scipy import stats as _st

    def mdes(n, alpha=0.05, power=0.80):
        tc = _st.t.ppf(1 - alpha / 2, n - 1)
        tb = _st.t.ppf(power, n - 1)
        return (tc + tb) / np.sqrt(n)

    sd = S["arms"]["pr"]["sd"]
    curve_db = np.array([mdes(n) * sd for n in ns])

    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    ax.plot(ns, curve_db, color=C["pr"], lw=2.2, zorder=3,
            label="MDES (80% power, paired t)")

    # 观测到的两个效应，作为水平线对比
    c_pos = S["comparisons"]["pr-fixed"]
    c_null = S["comparisons"]["unconstrained-fixed"]
    ax.axhline(abs(c_pos["delta"]), color=C["unconstrained"], lw=2,
               ls=(0, (5, 3)), zorder=2)
    ax.annotate("observed PR benefit  +%.3f dB" % abs(c_pos["delta"]),
                (ns[-1], abs(c_pos["delta"])), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=C["unconstrained"])
    ax.axhline(abs(c_null["delta"]), color=MUTED, lw=2, ls=(0, (5, 3)), zorder=2)
    # 标签放**左下**：右侧曲线末端会压到这条线，而左侧该高度是空的
    ax.annotate("observed null gap  %.3f dB" % abs(c_null["delta"]),
                (ns[0], abs(c_null["delta"])), xytext=(4, 7),
                textcoords="offset points", ha="left", fontsize=7.5, color=INK2)

    # 当前 n 的位置
    for n_cur, col in ((5, C["pr"]), (3, C["pr"])):
        ax.scatter([n_cur], [mdes(n_cur) * sd], s=70, color=col, zorder=5,
                   edgecolor=SURFACE, linewidth=1.6)
    ax.annotate("n = 5 (S1)\nMDES = %.3f dB" % (mdes(5) * sd),
                (5, mdes(5) * sd), xytext=(10, 14), textcoords="offset points",
                fontsize=7.5, color=INK, linespacing=1.35)
    ax.annotate("n = 3 (S2)", (3, mdes(3) * sd), xytext=(8, 20),
                textcoords="offset points", fontsize=7.5, color=INK2)

    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    # ⚠️ 对数轴的**次刻度标签**默认是开的，会冒出 "4×10⁰" / "3×10¹" 这类噪声
    #    （实测踩过）。主刻度已显式指定，次刻度标签必须关掉。
    ax.minorticks_off()
    ax.set_xlabel("Number of training seeds (n)")
    ax.set_ylabel("Minimum detectable Δ PSNR (dB)")
    ax.set_title("What this experiment can and cannot detect", loc="left", pad=8)
    despine(ax)
    ax.set_ylim(0, max(curve_db.max() * 0.55, abs(c_pos["delta"]) * 1.6))
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ====================================================== fig9 逐患者一致性
def fig9_perpatient(R, out):
    """逐患者分解 —— 效应是否在两个患者上同向。

    测试集只有 2 个患者，故这既是**稳健性证据**也是**局限的直观呈现**。
    """
    S = R["splits"]["S1"]
    pats = ["L506", "L067"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    ax = axes[0]
    x = np.arange(len(pats))
    w = 0.26
    for i, arm in enumerate(ORDER):
        vals = [S["arms"][arm]["per_patient"][p]["over_seeds_mean"] for p in pats]
        sds = [S["arms"][arm]["per_patient"][p]["over_seeds_sd_of_seedmeans"] for p in pats]
        ax.bar(x + (i - 1) * w, vals, w, color=C[arm], label=LAB[arm],
               edgecolor=SURFACE, linewidth=2, yerr=sds, capsize=2.5,
               error_kw=dict(ecolor=INK2, lw=1.1, zorder=4))
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.annotate("%.2f" % v, (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=7,
                        color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["L506 (n=211)", "L067 (n=224)"], fontsize=8.5)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) Effect direction is consistent across both test patients",
                 loc="left", pad=8)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.set_ylim(28.8, 33.0)

    # ---- (b) 每患者 identity 地板，显示患者间差异远大于臂间差异 ----
    ax = axes[1]
    floors = {"L506": 29.2489, "L067": 26.5576}
    best = {p: max(S["arms"][a]["per_patient"][p]["over_seeds_mean"] for a in ORDER)
            for p in pats}
    ax.bar(x - 0.16, [floors[p] for p in pats], 0.32, color=MUTED,
           label="Identity floor", edgecolor=SURFACE, linewidth=2)
    ax.bar(x + 0.16, [best[p] for p in pats], 0.32, color=C["pr"],
           label="Best arm", edgecolor=SURFACE, linewidth=2)
    for i, p in enumerate(pats):
        ax.annotate("%.2f" % floors[p], (i - 0.16, floors[p]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK)
        ax.annotate("%.2f" % best[p], (i + 0.16, best[p]), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=7.5, color=INK)
    ax.annotate("", xy=(0.5, 29.25), xytext=(0.5, 26.56),
                arrowprops=dict(arrowstyle="<->", color=C["unconstrained"], lw=1.6))
    ax.annotate("2.69 dB\npatient gap", (0.55, 27.7), fontsize=7.5,
                color=C["unconstrained"], linespacing=1.3)
    ax.set_xticks(x)
    ax.set_xticklabels(["L506", "L067"], fontsize=8.5)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(b) Patient gap vs. any effect we report", loc="left", pad=8)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.set_ylim(24.5, 33.5)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ============================================================== main
def main():
    R = load()
    os.makedirs(FIGDIR, exist_ok=True)
    print("生成扩展图（配色经 dataviz 校验器实测通过）")
    fig5_distribution(R, os.path.join(FIGDIR, "fig5_slice_dist.png"))
    fig6_convergence(R, os.path.join(FIGDIR, "fig6_convergence.png"))
    fig7_forest(R, os.path.join(FIGDIR, "fig7_forest.png"))
    fig8_power(R, os.path.join(FIGDIR, "fig8_power.png"))
    fig9_perpatient(R, os.path.join(FIGDIR, "fig9_perpatient.png"))


if __name__ == "__main__":
    main()
