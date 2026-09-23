"""新实验的图：E1（容量）、E3（初始化距离）、以及 bound↔drift 的机制图。

产出
================================================================
    fig15_e1_capacity.png   变换收益 vs 头容量（E1，**否定结果**）
    fig16_e3_initdrift.png  初始化距离 -> 训练后 drift（E3）
    fig17_bound_drift.png   bound -> drift -> PSNR 的机制链条

配色沿用已通过 dataviz 校验器的 #2E5EAA / #D1495B / #7B52AB。
"""

from __future__ import annotations

import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIGDIR = os.path.join(HERE, "..", "paper", "figures")
sys.path.insert(0, ROOT)

C = {"pr": "#2E5EAA", "unconstrained": "#D1495B", "fixed": "#7B52AB"}
INK, INK2, MUTED, GRID, SURFACE = "#1a1a19", "#55534d", "#8a8880", "#e3e2dd", "#fcfcfb"

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


def psnr(tag):
    p = os.path.join(HERE, "runs", tag, "results.json")
    if not os.path.exists(p):
        return None
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)["test_mean"]["PSNR"]


# ============================================================ fig15 E1
def fig15(out):
    """变换收益 vs 头容量。

    ⚠️ 这是**否定结果**（假设是"头越小增益越大"，实测方向相反）。
       图要如实呈现，包括端点档只有 n=3 这个限制。
    """
    caps = [(1, 94, "94"), (2, 2159, "2,159"), (3, 18399, "18,399")]
    rows = []
    for nc, npar, lab in caps:
        if nc == 2:
            pr = [psnr(f"fix_pr_s{s}") for s in range(10)]
            fx = [psnr(f"fix_fixed_s{s}") for s in range(10)]
            un = [psnr(f"fix_unconstrained_s{s}") for s in range(10)]
            n = 10
        else:
            pr = [psnr(f"e1_nc{nc}_pr_s{s}") for s in range(3)]
            fx = [psnr(f"e1_nc{nc}_fixed_s{s}") for s in range(3)]
            un = [psnr(f"e1_nc{nc}_unconstrained_s{s}") for s in range(3)]
            n = 3
        pr = np.array([x for x in pr if x is not None])
        fx = np.array([x for x in fx if x is not None])
        un = np.array([x for x in un if x is not None])
        if len(pr) == 0:
            continue
        m = min(len(pr), len(fx))
        from scipy import stats
        d = pr[:m] - fx[:m]
        t, p = stats.ttest_rel(pr[:m], fx[:m])
        rows.append({"npar": npar, "lab": lab, "pr": pr.mean(), "fx": fx.mean(),
                     "un": un.mean() if len(un) else np.nan,
                     "delta": d.mean(), "p": p, "n": n})

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    xs = np.arange(len(rows))
    w = 0.26
    for i, (key, col) in enumerate([("pr", C["pr"]), ("fx", C["fixed"]),
                                    ("un", C["unconstrained"])]):
        vals = [r[key] for r in rows]
        ax.bar(xs + (i - 1) * w, vals, w, color=col,
               label={"pr": "PR-LWT", "fx": "Fixed Haar",
                      "un": "Unconstrained"}[key],
               edgecolor=SURFACE, linewidth=1.8)
        for xi, v in zip(xs + (i - 1) * w, vals):
            ax.annotate("%.2f" % v, (xi, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    # 在每档上方标出 Δ(pr − fixed) 与其显著性
    for xi, r in zip(xs, rows):
        top = max(r["pr"], r["fx"], r["un"] if not np.isnan(r["un"]) else 0)
        star = ("***" if r["p"] < 0.001 else "**" if r["p"] < 0.01
                else "*" if r["p"] < 0.05 else "n.s.")
        ax.annotate("Δ=%+.3f\n(n=%d, %s)" % (r["delta"], r["n"], star),
                    (xi, top), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7, color=INK, linespacing=1.3)

    ax.set_xticks(xs)
    ax.set_xticklabels(["%s\n%s params" % (r["lab"], r["lab"]) for r in rows], fontsize=8)
    ax.set_ylabel("TEST PSNR (dB)")
    ax.set_title("The PR advantage does not shrink as the heads grow  "
                 "(hypothesis refuted)", loc="left", pad=8)
    ax.set_ylim(28.4, 31.9)
    ax.grid(axis="x", visible=False)
    despine(ax)
    ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ============================================================ fig16 E3
def fig16(out):
    """初始化距离 -> 训练后 drift。

    两段必须**分开画**：d<=2 是被拉向工作点的可恢复区间；
    d=4 起点前向已崩、训练完全无法自救。混在一起会被 d=4 带偏。
    """
    import torch
    from models.denoiser import PRWaveletDenoiser

    BOUND = 8.0
    levels = [(0.0, "d00"), (0.5, "d05"), (1.0, "d10"), (2.0, "d20"), (4.0, "d40")]
    rows = []
    for d, tag in levels:
        dr, ps = [], []
        for s in range(3):
            ck = os.path.join(HERE, "runs", f"e3_{tag}_s{s}", "best.pth")
            if not os.path.exists(ck):
                continue
            st = torch.load(ck, map_location="cpu", weights_only=False)
            m = PRWaveletDenoiser(bound=BOUND, init_drift=d)
            m.load_state_dict(st.get("model") or st.get("model_state_dict"))
            dr.append(m.wavelet.max_drift())
            x = psnr(f"e3_{tag}_s{s}")
            if x is not None:
                ps.append(x)
        if dr:
            rows.append({"d": d, "final": float(np.mean(dr)),
                         "psnr": float(np.mean(ps)) if ps else np.nan})

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # ---- (a) init -> final drift ----
    ax = axes[0]
    ax.plot([0, 4.5], [0, 4.5], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("no change", (3.9, 3.9), xytext=(-6, 8), textcoords="offset points",
                fontsize=7, color=INK2, ha="right")
    ok = [r for r in rows if r["d"] <= 2.0]
    bad = [r for r in rows if r["d"] > 2.0]
    if ok:
        ax.plot([r["d"] for r in ok], [r["final"] for r in ok],
                color=C["pr"], lw=2.2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                label="Recoverable")
    if bad:
        ax.plot([r["d"] for r in bad], [r["final"] for r in bad],
                color=C["unconstrained"], lw=2.2, marker="X", ms=10,
                markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4,
                label="Stuck (forward already broken)")
    ax.axhspan(0.55, 0.85, color=C["pr"], alpha=0.10, zorder=0)
    ax.annotate("preferred drift ≈ 0.73", (0.1, 0.80), fontsize=7.5, color=INK2)
    ax.set_xlabel("Initial drift from Haar")
    ax.set_ylabel("Drift after training")
    ax.set_title("(a) Training finds a preferred operating point", loc="left", pad=8)
    despine(ax)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(-0.2, 4.6)

    # ---- (b) init -> PSNR ----
    ax = axes[1]
    xs = [r["d"] for r in rows]
    ys = [r["psnr"] for r in rows]
    cols = [C["pr"] if r["d"] <= 2.0 else C["unconstrained"] for r in rows]
    ax.bar(xs, ys, 0.55, color=cols, edgecolor=SURFACE, linewidth=2)
    for x, y in zip(xs, ys):
        ax.annotate("%.2f" % y, (x, y), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK)
    ax.set_xlabel("Initial drift from Haar")
    ax.set_ylabel("TEST PSNR (dB)")
    ax.set_title("(b) Starting broken is unrecoverable", loc="left", pad=8)
    ax.set_ylim(0, 34)
    ax.grid(axis="x", visible=False)
    despine(ax)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


# ======================================================= fig17 bound->drift
def fig17(out):
    """bound -> 可达 drift -> PSNR 的机制链条。

    这是把 E3 与 bound 扫描接起来的一张图：
    bound 限定优化能搜到多远，drift 随之上升，PSNR 跟着 drift 走。
    """
    import torch
    from models.denoiser import PRWaveletDenoiser

    pairs = [(0.5, "fix_pr"), (1.0, "fix_bnd10"), (2.0, "fix_bnd20"),
             (4.0, "fix_bnd40"), (8.0, "fix_bnd80")]
    rows = []
    for b, tag in pairs:
        dr, ps = [], []
        for s in range(3):
            ck = os.path.join(HERE, "runs", f"{tag}_s{s}", "best.pth")
            if not os.path.exists(ck):
                continue
            st = torch.load(ck, map_location="cpu", weights_only=False)
            m = PRWaveletDenoiser(bound=b)
            m.load_state_dict(st.get("model") or st.get("model_state_dict"))
            dr.append(m.wavelet.max_drift())
            x = psnr(f"{tag}_s{s}")
            if x is not None:
                ps.append(x)
        if dr and ps:
            rows.append({"b": b, "drift": float(np.mean(dr)), "psnr": float(np.mean(ps))})

    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    # x 轴 = 实际 drift，不是 bound —— 这样 E3 的偏好点能直接叠上去
    xs = [r["drift"] for r in rows]
    ys = [r["psnr"] for r in rows]
    ax.plot(xs, ys, color=C["pr"], lw=2.2, marker="o", ms=8,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    for r in rows:
        ax.annotate("bound=%g" % r["b"], (r["drift"], r["psnr"]),
                    xytext=(0, -16), textcoords="offset points", ha="center",
                    fontsize=7, color=MUTED)
    # E3 独立测出的偏好点
    ax.axvline(0.73, color=C["unconstrained"], lw=1.8, ls=(0, (5, 3)), zorder=2)
    ax.annotate("preferred drift ≈ 0.73\n(measured independently in E3)",
                (0.73, min(ys)), xytext=(6, 10), textcoords="offset points",
                fontsize=7.5, color=C["unconstrained"], linespacing=1.3)
    ax.set_xlabel("Drift reached after training")
    ax.set_ylabel("TEST PSNR (dB)")
    # ⚠️ 标题必须克制：曲线峰值在 drift≈0.65（bound=4），而 E3 独立测出的偏好点是
    #    0.73 —— 两者接近但**不重合**（bound=8 冲到 0.78 反而略降）。
    #    能站住的是"默认 0.5 把 drift 压在平台之下"，不是"到达了偏好点"。
    ax.set_title("The default bound confines training below the plateau",
                 loc="left", pad=8)
    despine(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ->", out)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("生成新实验图表")
    fig15(os.path.join(FIGDIR, "fig15_e1_capacity.png"))
    fig16(os.path.join(FIGDIR, "fig16_e3_initdrift.png"))
    fig17(os.path.join(FIGDIR, "fig17_bound_drift.png"))


if __name__ == "__main__":
    main()
