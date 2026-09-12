"""
在给定划分上评价**平凡基线**，作为训练模型的对照下界。

为什么必须做：任何融合方法若要声称有效性，至少要打败
  - Average   : 0.5·(CT + MRI)   —— 零参数
  - CT only   : 直接输出 CT
  - MRI only  : 直接输出 MRI

实测（2026-09-12）发现：在 MI / CC / PSNR / VIF 上，Average **击败**了训练模型，
且差距（MI 42%）远大于模型之间的差距。故这组基线是论文中最关键的一张对照表。

⚠️ 注意指标性质分两类：
  - **参考型**（MI / CC / PSNR / VIF / SSIM / SCD / Qabf）：衡量与源图的关系
  - **无参考型**（EN / SD / SF / AG）：只衡量输出自身的锐度与信息量
  一个偏亮的、过度锐化的输出可以轻松抬高无参考型指标而不改善融合质量。

用法：
    python experiments/evaluate_trivial_baselines.py --split splits/ct_mri_case_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from utils.metrics import evaluate_pair  # noqa: E402

METRIC_KEYS = ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]
CSV_NAME = "data/manifest_ct_mri.csv"


def load_u8(p):
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "ct_mri_case_v1.json"))
    ap.add_argument("--split-key", default="splits",
                    choices=["splits", "slice_level_random"])
    ap.add_argument("--which", default="test", choices=["test", "val", "train"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.split, encoding="utf-8") as f:
        sp = json.load(f)
    block = (sp["slice_level_random"]["splits"] if args.split_key == "slice_level_random"
             else sp["splits"])
    ids = block[args.which]

    index = {}
    with open(os.path.join(ROOT, CSV_NAME), encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            index[row["pair_id"]] = (os.path.join(ROOT, row["path_a"]),
                                     os.path.join(ROOT, row["path_b"]))

    results = {}
    for name in ["Average", "CT_only", "MRI_only"]:
        rows = []
        for i in ids:
            ct, mri = load_u8(index[i][0]), load_u8(index[i][1])
            if name == "Average":
                fused = np.clip(np.rint((ct.astype(np.float64) + mri) / 2), 0, 255).astype(np.uint8)
            elif name == "CT_only":
                fused = ct
            else:
                fused = mri
            m = evaluate_pair(fused, ct, mri)
            rows.append({k: float(m[k]) for k in METRIC_KEYS})
        results[name] = {"mean": {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS},
                         "per_image": rows}

    print("=" * 100)
    print(f"平凡基线 —— {args.which} 集，{len(ids)} 对（{args.split_key}）")
    print("=" * 100)
    print(f"  {'方法':<12}" + "".join(f"{k:>9}" for k in METRIC_KEYS))
    print("  " + "-" * 96)
    for name, r in results.items():
        print(f"  {name:<12}" + "".join(f"{r['mean'][k]:>9.4f}" for k in METRIC_KEYS))

    print()
    print("  指标性质：EN/SD/SF 为**无参考型**（只看输出自身）；")
    print("            MI/CC/PSNR/SSIM/VIF/SCD/Qabf 为**参考型**（衡量与源图的关系）。")
    print("  ⚠️ 一个过度锐化或偏亮的输出可以抬高无参考型指标而不改善融合质量。")

    if args.out:
        p = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
