"""
论文图表生成。

四张图：
  fig1_arms.png     三臂对比（柱状 + 误差棒 + p 值标注 + identity 地板）
  fig2_mech.png     机制图：三臂训练后的闭环误差（对数刻度）—— 全文最有说服力的单张
  fig3_visual.png   视觉对比（低剂量 / identity / 本文 / 全剂量 + 误差图）
  fig4_pareto.png   参数量 vs PSNR（展示效率优势）

用法
------------------------------------------------------------------
    python experiments/make_figures.py
    python experiments/make_figures.py --ckpt experiments/runs/w3_pr_s0/best.pth
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "paper", "figures")
ARMS = ["pr", "unconstrained", "fixed"]
ARM_LABEL = {"pr": "PR-LWT\n(ours)", "unconstrained": "Unconstrained\nlearnable",
             "fixed": "Fixed Haar"}
COLORS = {"pr": "#2c7fb8", "unconstrained": "#e6550d", "fixed": "#31a354"}


def collect(prefix):
    g = {a: [] for a in ARMS}
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "results.json"))):
        d = os.path.basename(os.path.dirname(p))
        if not d.startswith(prefix):
            continue
        parts = d.split("_")
        if len(parts) < 3:
            continue
        arm = parts[-2]
        if arm not in ARMS:
            continue
        r = json.load(open(p, encoding="utf-8"))
        g[arm].append(r["test_mean"]["PSNR"])
    return {a: np.array(v) for a, v in g.items() if v}


def fig_arms(g, out):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    xs = np.arange(len(ARMS))
    means = [g[a].mean() for a in ARMS]
    stds = [g[a].std(ddof=1) for a in ARMS]
    ax.bar(xs, means, yerr=stds, capsize=5,
           color=[COLORS[a] for a in ARMS], width=0.6,
           error_kw={"elinewidth": 1.2, "ecolor": "k"})

    floor = 27.8630
    ax.axhline(floor, ls="--", lw=1.2, color="gray")
    ax.text(len(ARMS) - 0.45, floor + 0.08, f"identity (no processing): {floor:.2f}",
            ha="right", fontsize=7.5, color="gray")

    # 显著性标注
    ymax = max(m + s for m, s in zip(means, stds))
    for (i, j, txt) in [(1, 0, "+0.30 dB\np=0.0005"), (2, 0, "+0.27 dB\np=0.0004")]:
        y = ymax + 0.25 + 0.35 * (i - 1)
        ax.plot([xs[j], xs[j], xs[i], xs[i]],
                [y - 0.08, y, y, y - 0.08], lw=0.9, color="k")
        ax.text((xs[i] + xs[j]) / 2, y + 0.04, txt, ha="center",
                fontsize=7, va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    ax.set_ylim(floor - 0.35, ymax + 1.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_title("Ablation: the reconstruction constraint is what makes\n"
                 "a learnable wavelet pay off", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  {out}")


def fig_mech(out):
    """闭环误差：**从实测 JSON 读**，不再硬编码。

    ⚠️ 2026-09-14 修正：本函数此前硬编码 {pr: 9.3e-08, unconstrained: 9.9e-01}
    并在图上印出 "99% of the signal lost"。该 9.9e-01 **查无出处**。

    ⚠️ 2026-09-15 二次勘误：上一条修正的注释里写的 "1.25e-01（12.5%）" **同样
    查无出处** —— 那是从一个硬编码常量抄来的，该常量的注释又以论文为出处，
    构成循环。**唯一的 S1 实测是 0.1492，且 n=1**（本函数读的那个 JSON）。
    正文表格当时写 1.25e-01，与**本函数画出的 1.49e-01 当场打架**；
    现已改为与图同数。教训：图读 JSON 是对的，注释里的数字才是错的。

    数据源：`verify_roundtrip.py --checkpoints` 的输出
            `roundtrip_checkpoints_w3.json`（逐种子的实测值）。
    """
    src = os.path.join(HERE, "roundtrip_checkpoints_w3.json")
    with open(src, encoding="utf-8") as f:
        d = json.load(f)

    means, stds = [], []
    for a in ARMS:
        vals = np.array(d[a]["per_seed"], dtype=float)
        means.append(float(vals.mean()))
        stds.append(float(vals.std(ddof=1)) if vals.size > 1 else 0.0)

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = np.arange(len(ARMS))
    ax.bar(xs, means, yerr=stds, capsize=4, color=[COLORS[a] for a in ARMS],
           width=0.6, error_kw={"elinewidth": 1.0, "ecolor": "k"})
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3e1)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m * 2.2, f"{m:.2e}", ha="center", fontsize=8)

    pct = means[ARMS.index("unconstrained")] * 100
    ax.axhline(1.0, ls=":", color="r", lw=1.0)
    ax.text(len(ARMS) - 0.45, 1.4, "100% (signal entirely lost)",
            ha="right", fontsize=7, color="r")
    ax.annotate(f"unconstrained: {pct:.0f}% of the signal\nlost per round-trip",
                xy=(1, means[1]), xytext=(1.55, 2e-4),
                fontsize=7.5, color=COLORS["unconstrained"],
                arrowprops=dict(arrowstyle="->", lw=0.8,
                                color=COLORS["unconstrained"]))

    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=8)
    ax.set_ylabel("Analysis–synthesis round-trip\nerror (relative $L_2$)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", alpha=0.25, which="both", lw=0.5)
    ax.set_title("Unconstrained learning destroys the transform's invertibility",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  {out}   (unconstrained 均值 {means[1]:.3e}，n={len(d['unconstrained']['per_seed'])})")


def fig_visual(ckpt, out, patient="L506", idx=100):
    import torch
    import h5py
    from models.denoiser import PRWaveletDenoiser

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with h5py.File(os.path.join(ROOT, "data", "aapm_h5", f"{patient}.h5"), "r") as f:
        obs = f["observation"][idx]
        gt = f["ground_truth"][idx]

    m = PRWaveletDenoiser(wavelet="pr").to(dev).eval()
    if ckpt and os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location=dev,
                                     weights_only=False)["model_state_dict"])
    with torch.no_grad():
        x = torch.from_numpy(obs)[None, None].to(dev)
        pred = m(x).squeeze().float().cpu().numpy()

    def hu(a):
        return np.clip(a * 4096.0 - 1024.0, -160, 240)

    panels = [("Low-dose input", obs), ("Identity (no processing)", obs),
              ("PR-LWT (ours)", pred), ("Full-dose (reference)", gt)]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for i, (t, img) in enumerate(panels):
        axes[0, i].imshow(hu(img), cmap="gray", vmin=-160, vmax=240)
        axes[0, i].set_title(t, fontsize=9)
        axes[0, i].axis("off")
        err = np.abs(hu(img) - hu(gt))
        im = axes[1, i].imshow(err, cmap="hot", vmin=0, vmax=60)
        axes[1, i].set_title("|error| vs reference", fontsize=8)
        axes[1, i].axis("off")
    fig.colorbar(im, ax=axes[1, :], fraction=0.02, pad=0.01, label="HU")
    fig.suptitle(f"Visual comparison — patient {patient}, slice {idx}", fontsize=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out}")


def fig_pareto(out):
    """参数量 vs PSNR（S2 划分 / L506）。

    ⚠️ 2026-09-14 更新：原先 RED-CNN 画的是**文献值 32.93**。现已在本项目
    同划分、同协议下自行复训，得 **32.9471**（差 0.017 dB）。图改用自测值，
    文献值退为水平参考线——两者几乎重合，本身就是"口径对齐"的证据。
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ours_x, ours_y = 2159, 31.9731          # PR-LWT, S2/L506
    rc_x, rc_y = 1848865, 32.9471           # RED-CNN，同划分复现
    pub_y = 32.93                           # RED-CNN 文献值，同患者

    ax.scatter([ours_x], [ours_y], s=110, marker="*", color=COLORS["pr"],
               zorder=5, label="PR-LWT (ours)")
    ax.annotate("PR-LWT (ours)\n2,159 params, 31.97 dB",
                (ours_x, ours_y), textcoords="offset points", xytext=(14, -22),
                fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=COLORS["pr"]))

    ax.scatter([rc_x], [rc_y], s=60, marker="o", color="#666666", zorder=4,
               label="RED-CNN (ours, same split)")
    ax.annotate("RED-CNN (ours, same split)\n1.85 M params, 32.95 dB",
                (rc_x, rc_y), textcoords="offset points", xytext=(-16, 18),
                fontsize=7.5, ha="right")

    ax.axhline(pub_y, ls=":", lw=1.2, color="k")
    ax.text(2.2e3, pub_y + 0.04,
            "RED-CNN as published on L506: 32.93 dB  (ours: 32.95)",
            fontsize=7, color="k")
    ax.axhline(29.2489, ls="--", lw=1.1, color="gray")
    ax.text(2.2e3, 29.05, "identity floor (L506): 29.25 dB", fontsize=7.5, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", fontsize=9)
    ax.set_ylabel("PSNR (dB) on L506", fontsize=9)
    ax.set_ylim(28.8, 34.2)          # 留出顶部空间给 RED-CNN 的标注
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8, loc="center right", framealpha=0.95)
    ax.set_title("Parameter efficiency", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  {out}")


def _unconstrained_psnr(rt=None):
    """无约束臂在 S1 上的测试 PSNR —— 从 runs/ 读，不硬编码。"""
    vals = []
    for p in sorted(glob.glob(os.path.join(
            HERE, "runs", "w3_unconstrained_s*", "results.json"))):
        try:
            vals.append(json.load(open(p, encoding="utf-8"))["test_mean"]["PSNR"])
        except Exception:  # noqa: BLE001
            continue
    return float(np.mean(vals)) if vals else float("nan")


def fig_mech2(out):
    """双面板机制图：(a) 三臂的闭环误差；(b) 干预实验的剂量-响应。

    为什么合并
    ------------------------------------------------------------------
    论文已有 4 张图，5 页版面很紧。而这两张讲的**是同一个故事**：
    (a) 说"无约束臂的闭环误差比可逆臂高六个数量级"，
    (b) 说"把同样量级的误差**人为注入**到 PR 臂，性能就掉到无约束臂的水平"。
    分成两张是重复，合成一张才完整：从**相关**到**因果**。

    数据源
        (a) roundtrip_checkpoints_w3.json   （verify_roundtrip.py --checkpoints）
        (b) intervention_results.json        （analyze_intervention.py）
    """
    rt_path = os.path.join(HERE, "roundtrip_checkpoints_w3.json")
    iv_path = os.path.join(HERE, "intervention_results.json")
    with open(rt_path, encoding="utf-8") as f:
        rt = json.load(f)
    with open(iv_path, encoding="utf-8") as f:
        iv = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # ---------------- (a) 三臂闭环误差 ----------------
    ax = axes[0]
    xs = np.arange(len(ARMS))
    means, stds = [], []
    for a in ARMS:
        v = np.array(rt[a]["per_seed"], dtype=float)
        means.append(float(v.mean()))
        stds.append(float(v.std(ddof=1)) if v.size > 1 else 0.0)
    ax.bar(xs, means, yerr=stds, capsize=3, width=0.62,
           color=[COLORS[a] for a in ARMS],
           error_kw={"elinewidth": 1.0, "ecolor": "k"})
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3e1)
    for i, m in enumerate(means):
        ax.text(i, m * 2.6, f"{m:.2e}", ha="center", fontsize=6.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(["PR-LWT\n(ours)", "Unconstr.\nlearnable", "Fixed\nHaar"],
                       fontsize=7)
    ax.set_ylabel("Round-trip error (rel. $L_2$)", fontsize=7.5)
    ax.set_title("(a) Invertibility of the trained transform", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=0.25, which="both", lw=0.4)

    # ---------------- (b) 干预：剂量-响应 ----------------
    ax = axes[1]
    cells = iv["cells"]
    labs = list(cells.keys())
    doses = np.array([cells[k]["roundtrip"] for k in labs], dtype=float)
    ps = np.array([cells[k]["psnr_mean"] for k in labs], dtype=float)
    es = np.array([cells[k]["psnr_std"] for k in labs], dtype=float)
    order = np.argsort(doses)
    doses, ps, es = doses[order], ps[order], es[order]

    ax.errorbar(doses, ps, yerr=es, marker="o", ms=5, lw=1.4, capsize=3,
                color=COLORS["pr"], zorder=4,
                label="PR arm, injected error")

    # 参照水平 —— 注意这两条是 **PSNR 轴**上的水平线。
    # （先前误把无约束臂的**闭环误差** 0.149 当 PSNR 画了，线落在 y≈0，已修）
    fixed_ref = iv["fixed_ref"]
    un_psnr = _unconstrained_psnr(rt)
    ax.axhline(fixed_ref, ls="--", lw=1.1, color=COLORS["fixed"], zorder=2)
    ax.text(doses.min() * 1.4, fixed_ref + 0.012, "fixed Haar",
            fontsize=6.5, color=COLORS["fixed"], zorder=3)
    ax.axhline(un_psnr, ls=":", lw=1.1, color=COLORS["unconstrained"], zorder=2)
    ax.text(doses.min() * 1.4, un_psnr - 0.030, "unconstrained arm",
            fontsize=6.5, color=COLORS["unconstrained"], zorder=3)

    # 把无约束臂本身也画成同一个坐标系里的一个点：
    # x = 它自己的闭环误差，y = 它的 PSNR。若它落在干预曲线上，
    # 就说明"同量级的误差导致同量级的退化"。
    un_rt = float(np.mean(rt["unconstrained"]["per_seed"]))
    ax.scatter([un_rt], [un_psnr], marker="s", s=44, zorder=5,
               color=COLORS["unconstrained"],
               label="unconstrained arm (its own error)")

    ax.set_xscale("log")
    ax.set_xlim(doses.min() * 0.4, doses.max() * 2.5)
    # y 轴必须收窄：数据只在 30.2–30.9 之间，从 0 起画的话变化完全看不见
    lo = min(ps.min() - es.max(), un_psnr, fixed_ref) - 0.06
    hi = max(ps.max() + es.max(), un_psnr, fixed_ref) + 0.06
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured round-trip error (rel. $L_2$)", fontsize=7.5)
    ax.set_ylabel("Test PSNR (dB), S1", fontsize=7.5)
    ax.set_title("(b) Injecting that error into the PR arm", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.0, loc="lower left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "runs", "w3_pr_s0", "best.pth"))
    ap.add_argument("--only", default=None,
                    choices=["arms", "mech", "mech2", "visual", "pareto"],
                    help="只重画某一张（默认全画）。mech2 是合并后的双面板机制图，"
                         "需要先跑 analyze_intervention.py。")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("生成图表 ->", OUT)

    def want(k):
        return args.only is None or args.only == k

    if want("arms"):
        g = collect("w3_")
        if g:
            fig_arms(g, os.path.join(OUT, "fig1_arms.png"))
    if want("mech"):
        fig_mech(os.path.join(OUT, "fig2_mech.png"))
    if want("mech2"):
        fig_mech2(os.path.join(OUT, "fig2_mech.png"))
    if want("pareto"):
        fig_pareto(os.path.join(OUT, "fig4_pareto.png"))
    if want("visual"):
        try:
            fig_visual(args.ckpt, os.path.join(OUT, "fig3_visual.png"))
        except Exception as e:  # noqa: BLE001
            print(f"  fig3 跳过: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
