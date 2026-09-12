"""
按病例的 K 折交叉验证运行器。

为什么
------------------------------------------------------------------
单次划分只剩 2 个测试病例，导致：
  - 逐图 p 值是伪重复（20 张相邻脑切片来自同一病例，不是独立样本）
  - 有效独立样本数为 2，低方差指标（PSNR/SSIM）在该规模下测不出差异

K 折让**每个病例各被测一次**，从而在**病例层面**聚合，
得到 K×每折病例数 个独立测量（CT-MRI 为 10 个）。

聚合口径
------------------------------------------------------------------
先在**病例内**对切片取平均，再跨病例统计。
**不要把切片当独立样本**——同一折内 20 张相邻切片高度相关。

用法
------------------------------------------------------------------
    python experiments/run_cv.py --label my_config \
        --split splits/ct_mri_cv5.json --gate neutral_sigmoid --wavelet lifting \
        --gamma 1 --seeds 0 --epochs 50

    # 追加更多配置对比时，用不同 --label 再跑一次，然后：
    python experiments/run_cv.py --report-only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
METRIC_KEYS = ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]


def run_folds(args):
    """逐折调用 run_experiment.py。"""
    with open(os.path.join(ROOT, args.split), encoding="utf-8") as f:
        sp = json.load(f)
    n_folds = sp["n_folds"]

    for seed in [int(s) for s in args.seeds.split(",")]:
        for fold in range(n_folds):
            tag = f"{args.label}_f{fold}_s{seed}"
            cmd = [
                sys.executable, os.path.join(HERE, "run_experiment.py"),
                "--tag", tag, "--label", args.label,
                "--split", args.split, "--fold", str(fold),
                "--seed", str(seed), "--epochs", str(args.epochs),
                "--oversample", str(args.oversample),
                "--val-interval", str(args.val_interval),
                "--gate", args.gate, "--wavelet", args.wavelet,
                "--gamma", str(args.gamma), "--delta", str(args.delta),
                "--mid-channels", str(args.mid_channels),
            ]
            if args.edge_refine:
                cmd.append("--edge-refine")
            print(f"\n{'='*70}\n[{args.label}] 折 {fold}  种子 {seed}\n{'='*70}", flush=True)
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            tail = "\n".join((r.stdout or "").strip().splitlines()[-6:])
            print(tail, flush=True)
            if r.returncode != 0:
                print(f"⚠️ 折 {fold} 种子 {seed} 退出码 {r.returncode}", flush=True)
                print((r.stderr or "")[-1500:], flush=True)


def collect(label):
    """收集某 label 的所有折结果，按病例聚合。"""
    per_case = defaultdict(lambda: defaultdict(list))   # case -> metric -> [values]
    per_fold = defaultdict(list)
    n_runs = 0
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "results.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if r.get("tag", "").rsplit("_f", 1)[0] != label:
            continue
        n_runs += 1
        by_case = defaultdict(lambda: defaultdict(list))
        for row in r["test"]["per_image"]:
            c = int(row["id"]) // 1000
            for k in METRIC_KEYS:
                by_case[c][k].append(row[k])
        for c, d in by_case.items():
            fold = r["tag"].rsplit("_f", 1)[1].split("_")[0]
            per_fold[fold].append({k: float(np.mean(v)) for k, v in d.items()})
            for k in METRIC_KEYS:
                per_case[c][k].append(float(np.mean(d[k])))
    return per_case, per_fold, n_runs


def report(args):
    labels = args.report_labels.split(",") if args.report_labels else [args.label]
    all_res = {}
    for lb in labels:
        pc, pf, n = collect(lb)
        if not pc:
            print(f"（未找到 label={lb} 的结果）")
            continue
        all_res[lb] = (pc, pf, n)

    if not all_res:
        return

    print("\n" + "=" * 108)
    print("按病例交叉验证结果")
    print("=" * 108)

    for lb, (pc, pf, n) in all_res.items():
        cases = sorted(pc)
        print(f"\n【{lb}】  {n} 次运行，覆盖 {len(cases)} 个病例: {cases}")
        print(f"  {'病例':<6}" + "".join(f"{k:>9}" for k in METRIC_KEYS))
        print("  " + "-" * (6 + 9 * len(METRIC_KEYS)))
        for c in cases:
            print(f"  {c:<6}" + "".join(f"{np.mean(pc[c][k]):>9.4f}" for k in METRIC_KEYS))
        print("  " + "-" * (6 + 9 * len(METRIC_KEYS)))
        print(f"  {'每病例均值':<6}" )
        print(f"  {'mean':<6}" + "".join(
            f"{np.mean([np.mean(pc[c][k]) for c in cases]):>9.4f}" for k in METRIC_KEYS))
        print(f"  {'std':<6}" + "".join(
            f"{np.std([np.mean(pc[c][k]) for c in cases]):>9.4f}" for k in METRIC_KEYS))
        print(f"  ↑ 独立单元是**病例**（n={len(cases)}），不是切片。")

    if len(all_res) >= 2:
        print("\n" + "=" * 108)
        print("组间差值（病例级配对，正=后者更优）")
        print("=" * 108)
        keys = list(all_res)
        base = keys[0]
        pc0 = all_res[base][0]
        for lb in keys[1:]:
            pc1 = all_res[lb][0]
            common = sorted(set(pc0) & set(pc1))
            print(f"\n  【{lb}】 - 【{base}】   共同病例 n={len(common)}")
            print(f"  {'指标':<7}{base:>13}{lb:>13}{'差值':>11}{'病例级配对 t':>14}")
            print("  " + "-" * 58)
            for k in METRIC_KEYS:
                a = np.array([np.mean(pc0[c][k]) for c in common])
                b = np.array([np.mean(pc1[c][k]) for c in common])
                d = b - a
                # 病例级配对 t 统计量（n 小，仅作参考）
                sd = d.std(ddof=1) if len(d) > 1 else 0.0
                t = (d.mean() / (sd / np.sqrt(len(d)))) if sd > 1e-12 else 0.0
                print(f"  {k:<7}{a.mean():>13.4f}{b.mean():>13.4f}"
                      f"{d.mean():>+11.4f}{t:>14.2f}")

    print("\n⚠️  独立单元为病例（n=10）。n 仍偏小，t 值仅供参考，")
    print("    不应据此声称统计显著性。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None, help="实验条件标签（决定输出目录）")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report-labels", default=None, help="逗号分隔，要对比的多个 label")
    ap.add_argument("--split", default="splits/ct_mri_cv5.json")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--oversample", type=int, default=3)
    ap.add_argument("--val-interval", type=int, default=5)
    ap.add_argument("--gate", default="neutral_sigmoid")
    ap.add_argument("--wavelet", default="lifting")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--mid-channels", type=int, default=32)
    ap.add_argument("--edge-refine", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        if not args.label:
            ap.error("非 --report-only 时必须给 --label")
        run_folds(args)
    report(args)


if __name__ == "__main__":
    main()
