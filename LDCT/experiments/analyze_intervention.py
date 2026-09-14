"""
干预实验分析：向 PR 臂注入受控闭环误差后，性能是否随剂量单调退化。

要回答的问题
------------------------------------------------------------------
论文 §5.3 目前把一个**归因**说成机制：无约束臂丧失了可逆性（闭环误差 12–15%），
所以它的性能掉到固定 Haar 水平。但从没说"是**这个**误差导致的"。

本实验补上这个干预：固定其他一切，只向 PR 臂注入闭环误差 ε ∈ {6%, 12.5%, 25%}，
看性能如何变化。

判定标准（必须先写死，否则容易事后找解释）
------------------------------------------------------------------
  · **单调退化 + ε=12.5% 处落到固定 Haar 水平** → 机制得到因果支持
  · **不单调 / 无明显退化** → 归因是错的，§5.3 必须改写

⚠️ 剂量以**训练后实测**的闭环误差为准，不是设定值 ε。因为 U 在训练中会漂移，
   ε=0.222 训练后未必正好是 12.5%。

用法
    python experiments/analyze_intervention.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from models.denoiser import PRWaveletDenoiser     # noqa: E402

RUNS = os.path.join(HERE, "runs")

# 设定的注入值 -> (标签, run 前缀)
SEEDS = [0, 1, 2, 3, 4]      # 与 w3_pr_s0..s4 对照臂同种子，可配对

CELLS = [
    ("ε=0（对照）", "w3_pr", 0.0, SEEDS),
    ("ε=0.111", "mis111", 0.111, SEEDS),
    ("ε=0.222", "mis222", 0.222, SEEDS),
    ("ε=0.444", "mis444", 0.444, SEEDS),
]

# 固定 Haar 臂的参照值（S1，来自 w3_fixed_s*）
FIXED_REF = 30.5591


def probe(model, size=256, seed=7):
    """与 verify_roundtrip.py 完全相同的探针，保证与论文数字可比。

    注意：随机数必须在 CPU 上按固定种子生成后再搬到模型所在设备，
    否则同一种子在 CPU 与 CUDA 上生成的是**不同的图**，数字就不可比了。
    """
    g = torch.Generator().manual_seed(seed)          # 固定 CPU 生成器
    x = torch.rand(1, 1, size, size, generator=g)
    dev = next(model.parameters()).device
    return model.transform_roundtrip(x.to(dev))[1]


def load_psnr(prefix, seed):
    p = os.path.join(RUNS, f"{prefix}_s{seed}", "results.json")
    if not os.path.exists(p):
        return None, None
    r = json.load(open(p, encoding="utf-8"))
    return r["test_mean"]["PSNR"], r["test_mean"]["SSIM"]


def load_roundtrip(prefix, seed, dev):
    ck = os.path.join(RUNS, f"{prefix}_s{seed}", "best.pth")
    if not os.path.exists(ck):
        return None
    st = torch.load(ck, map_location=dev, weights_only=False)
    cfg = st["config"]
    if cfg.get("wavelet") != "pr":
        return None
    m = PRWaveletDenoiser(levels=cfg.get("levels", 2),
                          mid_ch=cfg.get("mid_ch", 16),
                          n_conv=cfg.get("n_conv", 2),
                          wavelet="pr",
                          synth_mismatch=cfg.get("synth_mismatch", 0.0)).to(dev)
    m.load_state_dict(st["model_state_dict"])
    m.eval()
    return probe(m)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print("干预实验：向 PR 臂注入受控闭环误差 —— 剂量-响应")
    print("=" * 88)
    print(f"  协议与主实验完全相同（epochs=30 patch=128 bs=8 lr=1e-3 val_interval=5，S1 划分）")
    print(f"  固定 Haar 臂参照值：{FIXED_REF:.4f} dB\n")

    print(f"  {'设置':<14}{'实测闭环':>12}{'PSNR 均值':>11}{'标准差':>9}"
          f"{'逐种子':>26}{'vs 对照':>10}")
    print("  " + "-" * 84)

    out = {}
    ref_psnr = None
    for label, prefix, eps, seeds in CELLS:
        psnrs, rts = [], []
        for s in seeds:
            p, _ = load_psnr(prefix, s)
            if p is None:
                continue
            psnrs.append(p)
            rt = load_roundtrip(prefix, s, dev)
            if rt is not None:
                rts.append(rt)
        if not psnrs:
            continue
        arr = np.array(psnrs)
        rt_mean = float(np.mean(rts)) if rts else float("nan")
        if ref_psnr is None:
            ref_psnr = arr.mean()
        delta = arr.mean() - ref_psnr
        seeds_str = " ".join(f"{v:.3f}" for v in arr)
        print(f"  {label:<14}{rt_mean:>12.3e}{arr.mean():>11.4f}"
              f"{arr.std(ddof=1):>9.4f}{seeds_str:>26}{delta:>+10.4f}")
        out[label] = {"eps": eps, "roundtrip": rt_mean,
                      "psnr_mean": float(arr.mean()),
                      "psnr_std": float(arr.std(ddof=1)),
                      "per_seed": psnrs, "n": len(arr)}

    print("  " + "-" * 84)
    print(f"  {'固定 Haar 参照':<14}{'—':>12}{FIXED_REF:>11.4f}"
          f"{'（来自 w3_fixed_s*）':>35}")

    # ---- 配对比较（同种子）----
    print("\n  配对比较（同种子配对，独立单元 = 种子）")
    print("  " + "-" * 84)
    ctrl = {s: load_psnr("w3_pr", s)[0] for s in SEEDS}
    for label, prefix, eps, seeds in CELLS[1:]:
        pair_a, pair_b = [], []
        for s in SEEDS:
            p = load_psnr(prefix, s)[0]
            if p is None or ctrl.get(s) is None:
                continue
            pair_a.append(ctrl[s])
            pair_b.append(p)
        if len(pair_a) < 2:
            continue
        d = np.array(pair_a) - np.array(pair_b)
        try:
            from scipy import stats
            t, pv = stats.ttest_rel(pair_a, pair_b)
            crit = stats.t.ppf(0.975, len(d) - 1)
            ci = (d.mean() - crit * d.std(ddof=1) / np.sqrt(len(d)),
                  d.mean() + crit * d.std(ddof=1) / np.sqrt(len(d)))
            print(f"  {label:<12} 相对 ε=0 掉了 {d.mean():+.4f} dB  "
                  f"t={t:5.2f}  p={pv:.4f}  95%CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]")
        except Exception:  # noqa: BLE001
            print(f"  {label:<12} 相对 ε=0 掉了 {d.mean():+.4f} dB  "
                  f"（n={len(d)}）")

    print("\n" + "=" * 88)
    print("【判定】（标准在脚本开头已写死，避免事后找解释）")
    print("=" * 88)
    keys = list(out.keys())
    ps = [out[k]["psnr_mean"] for k in keys]
    rts = [out[k]["roundtrip"] for k in keys]
    monotone = all(ps[i] >= ps[i + 1] for i in range(len(ps) - 1))
    print(f"  PSNR 随剂量单调不增：{'是' if monotone else '否'}")
    if len(ps) >= 2:
        print(f"  总降幅 ε=0 → ε=0.444：{ps[0] - ps[-1]:+.4f} dB")
    print(f"  最高剂量处相对固定 Haar 参照（{FIXED_REF:.4f}）："
          f"{ps[-1] - FIXED_REF:+.4f} dB")
    print()
    if monotone and (ps[0] - ps[-1]) > 0.15:
        print("  => 支持因果：注入闭环误差本身即导致性能退化。")
    elif not monotone:
        print("  => ⚠️ 不单调：机制归因**未获支持**，§5.3 必须改写。")
    else:
        print("  => ⚠️ 退化幅度过小：机制归因**证据不足**，需谨慎表述。")

    dst = os.path.join(HERE, "intervention_results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"cells": out, "fixed_ref": FIXED_REF,
                   "monotone": bool(monotone)}, f, ensure_ascii=False, indent=2)
    print(f"\n  已写入 {dst}")


if __name__ == "__main__":
    main()
