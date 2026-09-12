"""
汇总 experiments/runs/ 下的实验结果，输出对比表。

用法：
    python experiments/aggregate.py                      # 汇总全部
    python experiments/aggregate.py --group-by gate_type # 按门控类型分组
    python experiments/aggregate.py --filter smoke=0     # 排除 tag 含 smoke 的
    python experiments/aggregate.py --csv out.csv

跨种子报告均值 ± 标准差。**单种子结果不得用于任何结论**（见修订版 §四）。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
METRIC_KEYS = ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]


def load_runs(pattern="runs/*/results.json"):
    runs = []
    for p in sorted(glob.glob(os.path.join(HERE, pattern))):
        try:
            with open(p, encoding="utf-8") as f:
                r = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"  跳过 {p}: {e}", file=sys.stderr)
            continue
        cfgs = os.path.join(os.path.dirname(p), "config.json")
        if os.path.exists(cfgs):
            with open(cfgs, encoding="utf-8") as f:
                r["_config"] = json.load(f)
        runs.append(r)
    return runs


def main():
    ap = argparse.ArgumentParser(description="汇总实验结果")
    ap.add_argument("--group-by", default="gate_type",
                    choices=["gate_type", "learnable_dwt", "tag"])
    ap.add_argument("--split-name", default="test", choices=["test", "val"])
    ap.add_argument("--exclude", default="smoke",
                    help="逗号分隔的 tag 子串，匹配则排除")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    runs = load_runs()
    excl = [s for s in args.exclude.split(",") if s]
    runs = [r for r in runs if not any(e in r.get("tag", "") for e in excl)]
    if not runs:
        print("没有可汇总的实验结果")
        return

    groups = defaultdict(list)
    for r in runs:
        key = r.get(args.group_by)
        if key is None and "_config" in r:
            key = r["_config"].get(args.group_by)
        groups[str(key)].append(r)

    print("=" * 108)
    print(f"MGF-Net 实验结果汇总 —— 按 {args.group_by} 分组，报告 {args.split_name} 集")
    print("=" * 108)

    rows_out = []
    for g, rs in sorted(groups.items()):
        print(f"\n【{g}】  n={len(rs)} 次运行  种子={sorted(r['seed'] for r in rs)}")
        print(f"  {'指标':<8}{'均值':>10}{'标准差':>10}{'最小':>10}{'最大':>10}")
        print("  " + "-" * 48)
        for k in METRIC_KEYS:
            vals = [r[args.split_name]["mean"][k] for r in rs]
            import numpy as np
            print(f"  {k:<8}{np.mean(vals):>10.4f}{np.std(vals):>10.4f}"
                  f"{np.min(vals):>10.4f}{np.max(vals):>10.4f}")
            rows_out.append({"group": g, "metric": k, "mean": float(np.mean(vals)),
                             "std": float(np.std(vals)), "min": float(np.min(vals)),
                             "max": float(np.max(vals)), "n_runs": len(rs)})
        ep = [r.get("best_epoch") for r in rs]
        print(f"  {'best_epoch':<8} {ep}")

    # 组间差值表（仅当有两组时）
    if len(groups) == 2:
        (g1, r1), (g2, r2) = sorted(groups.items())
        import numpy as np
        print("\n" + "=" * 108)
        print(f"差值：{g2} - {g1}（正数表示 {g2} 更优）")
        print("=" * 108)
        print(f"  {'指标':<8}{g1:>14}{g2:>14}{'差值':>12}")
        print("  " + "-" * 50)
        for k in METRIC_KEYS:
            v1 = np.mean([r[args.split_name]["mean"][k] for r in r1])
            v2 = np.mean([r[args.split_name]["mean"][k] for r in r2])
            flag = "  <<<" if abs(v2 - v1) > 0.02 else ""
            print(f"  {k:<8}{v1:>14.4f}{v2:>14.4f}{v2 - v1:>+12.4f}{flag}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        print(f"\n已写入 {args.csv}")

    print("\n⚠️  提醒：本表基于 screening 划分，样本量极小，仅供机制筛查，不得写入论文。")


if __name__ == "__main__":
    main()
