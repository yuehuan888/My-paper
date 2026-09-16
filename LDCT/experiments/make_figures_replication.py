"""复现研究图表：把"独立重下数据后结论依然成立"这件事画出来。

产出（沿用 make_figures_ext.py 中已通过 dataviz 校验器的配色）：
    fig10_replication.png   三臂：原数据 vs 新数据，并排
    fig11_bound_curve.png   bound 剂量-响应曲线（贡献 3 从单点变曲线）
    fig12_effect_forest.png 效应量森林图：原 vs 新，含 MDES 带
    fig13_seed_conv.png     均值随种子数的收敛 + 95% CI 收窄
    fig14_power_n10.png     n=10 下的功效分析

配色沿用（实测通过的色盲友好组合）：
    #2E5EAA / #D1495B / #7B52AB   —— 相邻对 protan ΔE 14.3、常视觉 ΔE 20.0
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
RJSON = os.path.join(HERE, "analysis_replication.json")

C = {"pr": "#2E5EAA", "unconstrained": "#D1495B", "fixed": "#7B52AB"}
LAB = {"pr": "PR-LWT (ours)", "unconstrained": "Unconstrained", "fixed": "Fixed Haar"}
ORDER = ["pr", "unconstrained", "fixed"]

INK, INK2, MUTED = "#1a1a19", "#55534d", "#8a8880"
GRID, SURFACE = "#e3e2dd", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 300,
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load():
    with io.open(RJSON, encoding="utf-8") as f:
        return json.load(f)


# ============================================================ fig10 复现
def fig10(R, out):
    """三臂：原数据 vs 新数据。展示"独立重下数据后排序与差距都不变"。"""
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    x = np.arange(len(ORDER))
    w = 0.36
    for i, (src, alpha, hatch) in enumerate([("paper", 1.0, None), ("new", 0.45, "///")]):
        means, errs = [], []
        for a in ORDER:
            if src == "paper":
                m, s = R["paper"][a]
            else:
                m, s = R["arms"][a]["mean"], R["arms"][a]["sd"]
            means.append(m); errs.append(s)
        off = (-w / 2) if i == 0 else (w / 2)
        ax.bar(x + off, means, w, yerr=errs, capsize=3,
               color=[C[a] for a in ORDER], alpha=alpha, hatch=hatch,
               edgecolor=SURFACE, linewidth=1.6,
               error_kw=dict(ecolor=INK2, lw=1.1, zorder=5))
        for xi, m in zip(x + off, means):
            ax.annotate("%.2f" % m, (xi, m), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels([LAB[a] for a in ORDER], fontsize=8.5)
    ax.set_ylabel("TEST PSNR (dB)")
    ax.set_title("Independent replication: ranking and gaps are unchanged", loc="left", pad=8)
    ax.set_ylim(30.2, 31.15)
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.legend(handles=[
        Patch(facecolor=MUTED, label="Original data — 435 slices, n=5"),
        Patch(facecolor=MUTED, alpha=0.45, hatch="///", label="Re-acquired — 421 slices, n=10"),
    ], fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ========================================================= fig11 bound 曲线
def fig11(R, out):
    """bound 剂量-响应 —— 贡献 3 的核心图。

    形式：折线 + 误差棒。x 轴用对数刻度（bound 跨度 0.5→8）。
    **不画双 y 轴**。标出峰值与默认值。
    """
    B = R.get("bound", {})
    if not B:
        print("  (跳过 fig11：无 bound 数据)")
        return
    bs = sorted(float(k) for k in B)
    mean = np.array([B[str(b) if str(b) in B else ("%g" % b)]["mean"] for b in bs])
    sd = np.array([B[str(b) if str(b) in B else ("%g" % b)]["sd"] for b in bs])
    ns = [B[str(b) if str(b) in B else ("%g" % b)]["n_seeds"] for b in bs]

    # 名义峰值 vs 默认档的配对检验（用于标题措辞，必须算出真实 p 再写）
    from scipy import stats as _st
    k0, kb = str(bs[0]), str(bs[int(np.argmax(mean))])
    v0, vb = B.get(k0, {}).get("seeds", []), B.get(kb, {}).get("seeds", [])
    p_vs_default = (_st.ttest_rel(vb, v0).pvalue
                    if len(v0) == len(vb) and len(v0) > 1 else float("nan"))

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    # 默认 bound=0.5 的参照线
    ax.axhline(mean[0], color=MUTED, lw=1.4, ls=(0, (4, 3)), zorder=1)
    ax.annotate("default bound = 0.5", (bs[-1], mean[0]), xytext=(-4, -12),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)

    ax.errorbar(bs, mean, yerr=sd, color=C["pr"], lw=2.2, marker="o", ms=7,
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=1.5)

    ib = int(np.argmax(mean))
    ax.scatter([bs[ib]], [mean[ib]], s=150, facecolor="none",
               edgecolor=C["unconstrained"], linewidth=2, zorder=5)
    # ⚠️ 措辞必须克制：n=3 下**没有任何一档**与默认 0.5 显著不同
    #    （配对 t 检验 p 全部 > 0.13，CI 全部跨零）。
    #    所以只能说"名义最高"，不能说"最优"或"0.5 偏保守"。
    ax.annotate("nominal peak  %.4f dB\n(+%.3f, n.s.; p=%.2f)"
                % (mean[ib], mean[ib] - mean[0], p_vs_default),
                (bs[ib], mean[ib]), xytext=(12, -8), textcoords="offset points",
                fontsize=7.5, color=INK, linespacing=1.35)

    for b, m, n in zip(bs, mean, ns):
        ax.annotate("n=%d" % n, (b, m), xytext=(0, -16), textcoords="offset points",
                    ha="center", fontsize=6.5, color=MUTED)

    ax.set_xscale("log")
    ax.set_xticks(bs)
    ax.set_xticklabels(["%g" % b for b in bs])
    ax.minorticks_off()
    ax.set_xlabel(r"Tap bound  ($|$taps$| \leq$ init $+$ bound)")
    ax.set_ylabel("TEST PSNR (dB)")
    # 标题不能写成"0.5 偏保守"：实测 n=3 下没有任何一档与默认显著不同。
    # 能站住的结论是**稳健性**——只要界定住，性能对具体取值不敏感。
    ax.set_title("Performance is insensitive to the bound, once bounded",
                 loc="left", pad=8)
    despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ======================================================= fig12 效应量森林图
def fig12(R, out):
    """效应量森林图：原研究 vs 复现研究，含各自的 MDES 带。"""
    cmps = R.get("comparisons", {})
    if not cmps:
        print("  (跳过 fig12：无比较数据)")
        return
    keys = ["pr-unconstrained", "pr-fixed", "unconstrained-fixed"]
    keys = [k for k in keys if k in cmps]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    rows = []
    for i, k in enumerate(keys):
        c = cmps[k]
        rows.append(("replication", k, c["delta"], c["ci"], c["p"], c["mdes_dB"]))
        if c.get("paper"):
            pp = c["paper"]
            rows.append(("paper", k, pp["delta"], None, pp["p"], None))

    ypos, ylab = [], []
    y = 0
    for src, k, d, ci, p, mde in rows:
        ypos.append(y)
        a, b = k.split("-")
        tag = "new" if src == "replication" else "orig"
        ylab.append("%s  [%s]" % (a.replace("unconstrained", "unconstr")
                                  .replace("fixed", "fixed Haar"), tag))
        y += 1
        if src == "paper":
            y += 0.35

    for i, (src, k, d, ci, p, mde) in enumerate(rows):
        yy = ypos[i]
        sig = p < 0.05
        col = C[k.split("-")[0]] if sig else MUTED
        if mde:
            ax.barh(yy, 2 * mde, left=-mde, height=0.6, color=GRID, alpha=0.55,
                    zorder=1, linewidth=0)
        if ci:
            ax.plot(ci, [yy, yy], color=col, lw=2.2, solid_capstyle="round", zorder=3)
        ax.scatter([d], [yy], s=64, color=col, zorder=4,
                   edgecolor=SURFACE, linewidth=1.6)
        ax.annotate("%+.3f  (p=%.3f)" % (d, p), (max(ci[1] if ci else d, d), yy),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7.5, color=INK if sig else INK2)

    ax.axvline(0, color=INK2, lw=1.2, zorder=2)
    ax.set_yticks(ypos); ax.set_yticklabels(ylab, fontsize=7.5)
    ax.set_xlabel("Δ PSNR (dB)   [positive = left arm better]")
    ax.set_title("Effect sizes survive re-acquisition", loc="left", pad=8)
    ax.grid(axis="y", visible=False)
    despine(ax); ax.invert_yaxis()
    hi = max((r[3][1] if r[3] else r[2]) for r in rows)
    lo = min((r[3][0] if r[3] else r[2]) for r in rows)
    ax.set_xlim(lo - 0.08, hi + 0.30)
    ax.set_ylim(ypos[-1] + 0.6, ypos[0] - 0.8)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=2.2, marker="o", ms=6, ls="none",
               markeredgecolor=SURFACE, label="n.s. (p ≥ 0.05)"),
        Patch(facecolor=GRID, alpha=0.55, label="MDES band (n=10 replication)"),
    ], fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ==================================================== fig13 种子收敛
def fig13(R, out):
    """均值随种子数的收敛：说明 n=10 已经够稳。"""
    conv = R.get("convergence", {})
    if len(conv) < 3:
        print("  (跳过 fig13：收敛数据不足)")
        return
    ns = sorted(int(k) for k in conv)
    mean = np.array([conv[str(n)]["mean"] for n in ns])
    hw = np.array([conv[str(n)]["halfwidth"] for n in ns])

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(ns, mean, color=C["pr"], lw=2.2, marker="o", ms=4,
            markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)
    ax.fill_between(ns, mean - hw, mean + hw, color=C["pr"], alpha=0.16, linewidth=0)
    ax.axhline(mean[-1], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("final mean  %.4f dB" % mean[-1], (ns[-1], mean[-1]),
                xytext=(-6, 8), textcoords="offset points", ha="right",
                fontsize=7.5, color=INK2)
    # 参考：论文 n=5 的均值
    pm = R["paper"]["pr"][0]
    ax.axhline(pm, color=C["unconstrained"], lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.annotate("original study (n=5, 435 slices)  %.4f dB" % pm, (ns[0], pm),
                xytext=(4, 6), textcoords="offset points", fontsize=7.5,
                color=C["unconstrained"])
    ax.set_xlabel("Number of seeds (n)")
    ax.set_ylabel("Mean TEST PSNR (dB)")
    ax.set_title("The mean is stable well before n = 10", loc="left", pad=8)
    despine(ax)
    ax.legend(handles=[
        Line2D([], [], color=C["pr"], lw=2.2, marker="o", ms=4, label="Replication mean"),
        Patch(facecolor=C["pr"], alpha=0.16, label="95% CI half-width"),
    ], fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ==================================================== fig14 功效
def fig14(R, out):
    """n=10 下的功效：能检出多小的效应、观测效应落在哪。"""
    from scipy import stats as st
    c = R["comparisons"].get("pr-fixed")
    if not c:
        print("  (跳过 fig14：无 pr-fixed 比较)")
        return
    sd = R["arms"]["pr"]["sd"]

    def mdes(n, alpha=0.05, power=0.80):
        return float((st.t.ppf(1 - alpha / 2, n - 1) + st.t.ppf(power, n - 1)) / np.sqrt(n))

    ns = np.array([2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50])
    curve = np.array([mdes(n) * sd for n in ns])

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot(ns, curve, color=C["pr"], lw=2.2, zorder=3,
            label="MDES (80% power, paired t)")
    ax.axhline(abs(c["delta"]), color=C["unconstrained"], lw=2, ls=(0, (5, 3)), zorder=2)
    ax.annotate("observed PR benefit  +%.3f dB" % abs(c["delta"]),
                (ns[-1], abs(c["delta"])), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=7.5,
                color=C["unconstrained"])
    nul = R["comparisons"].get("unconstrained-fixed")
    if nul:
        ax.axhline(abs(nul["delta"]), color=MUTED, lw=2, ls=(0, (5, 3)), zorder=2)
        ax.annotate("observed null gap  %.3f dB" % abs(nul["delta"]),
                    (ns[0], abs(nul["delta"])), xytext=(4, 7),
                    textcoords="offset points", fontsize=7.5, color=INK2)
    for n_cur, col in ((10, C["pr"]), (5, C["pr"]), (3, C["pr"])):
        ax.scatter([n_cur], [mdes(n_cur) * sd], s=64, color=col, zorder=5,
                   edgecolor=SURFACE, linewidth=1.6)
    ax.annotate("n=10 (replication)\nMDES = %.3f dB" % (mdes(10) * sd),
                (10, mdes(10) * sd), xytext=(8, 16), textcoords="offset points",
                fontsize=7.5, color=INK, linespacing=1.35)
    ax.annotate("n=5 (original)", (5, mdes(5) * sd), xytext=(-8, 14),
                textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 10, 20, 50])
    ax.set_xticklabels(["2", "3", "5", "10", "20", "50"])
    ax.minorticks_off()
    ax.set_xlabel("Number of training seeds (n)")
    ax.set_ylabel("Minimum detectable Δ PSNR (dB)")
    ax.set_title("Detection floor at n = 10", loc="left", pad=8)
    despine(ax)
    ax.set_ylim(0, max(curve.max() * 0.5, abs(c["delta"]) * 1.7))
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


def main():
    R = load()
    os.makedirs(FIGDIR, exist_ok=True)
    print("生成复现研究图表（配色经 dataviz 校验器实测通过）")
    fig10(R, os.path.join(FIGDIR, "fig10_replication.png"))
    fig11(R, os.path.join(FIGDIR, "fig11_bound_curve.png"))
    fig12(R, os.path.join(FIGDIR, "fig12_effect_forest.png"))
    fig13(R, os.path.join(FIGDIR, "fig13_seed_conv.png"))
    fig14(R, os.path.join(FIGDIR, "fig14_power_n10.png"))


if __name__ == "__main__":
    main()
