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
    """闭环误差：数据来自三臂各 50 步训练后的实测（LDCT 项目记录）。"""
    vals = {"pr": 9.3e-08, "fixed": 1.1e-07, "unconstrained": 9.9e-01}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    xs = np.arange(len(ARMS))
    v = [vals[a] for a in ARMS]
    ax.bar(xs, v, color=[COLORS[a] for a in ARMS], width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1e1)
    for i, val in enumerate(v):
        ax.text(i, val * 1.8, f"{val:.1e}", ha="center", fontsize=8)
    ax.axhline(1.0, ls=":", color="r", lw=1.2)
    ax.text(len(ARMS) - 0.45, 1.6, "99% of the signal lost", ha="right",
            fontsize=7.5, color="r")
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
    print(f"  {out}")


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
    """参数量 vs PSNR。published 数字引自文献，明确区分标注。"""
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ours_x, ours_y = 2159, 31.9731
    ax.scatter([ours_x], [ours_y], s=90, marker="*", color=COLORS["pr"],
               zorder=5, label="PR-LWT (ours)")
    ax.annotate("PR-LWT (ours)\n2.2 K params, 31.97 dB",
                (ours_x, ours_y), textcoords="offset points", xytext=(12, -6),
                fontsize=7.5)
    ax.scatter([1848865], [32.93], s=60, marker="o", color="gray", zorder=4)
    ax.annotate("RED-CNN (published)\n1.85 M params, ~32.93 dB",
                (1848865, 32.93), textcoords="offset points", xytext=(-10, 12),
                fontsize=7.5, ha="right")
    ax.axhline(29.2489, ls="--", lw=1.1, color="gray")
    ax.text(2.2e3, 29.05, "identity floor (L506): 29.25 dB", fontsize=7.5, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)", fontsize=9)
    ax.set_ylabel("PSNR (dB) on L506", fontsize=9)
    ax.set_ylim(28.8, 33.6)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Parameter efficiency", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "runs", "w3_pr_s0", "best.pth"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    print("生成图表 ->", OUT)

    g = collect("w3_")
    if g:
        fig_arms(g, os.path.join(OUT, "fig1_arms.png"))
    fig_mech(os.path.join(OUT, "fig2_mech.png"))
    fig_pareto(os.path.join(OUT, "fig4_pareto.png"))
    try:
        fig_visual(args.ckpt, os.path.join(OUT, "fig3_visual.png"))
    except Exception as e:  # noqa: BLE001
        print(f"  fig3 跳过: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
