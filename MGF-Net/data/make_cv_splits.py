"""
按**病例**生成 K 折交叉验证划分。

为什么需要
------------------------------------------------------------------
单次划分只能用 2 个病例做测试（见 `splits/ct_mri_case_v1.json`），
导致：
  - 逐图 p 值是**伪重复**（40 张图来自 2 个病例，不是 40 个独立样本）
  - 有效独立样本数只有 2，PSNR/SSIM 这类低方差指标在该规模下**永远测不出差异**

按病例做 K 折可得到 **每个病例各被测一次**，从而：
  - 在病例层面聚合，得到 K×每折病例数 个独立测量
  - 每折内部仍是干净的病例隔离（训练集不含测试病例）

设计
------------------------------------------------------------------
CT-MRI 共 10 个病例，5 折 → 每折 2 个测试病例、1 个验证病例、7 个训练病例。
分配是**确定性的**（按病例号排序后依次分配），任何人可复现，无随机。

用法
------------------------------------------------------------------
    python data/make_cv_splits.py --modality CT-MRI --folds 5 --name ct_mri_cv5
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def make_folds(case_ids, n_folds: int):
    """把病例号确定性地分到 K 折。返回 [[case,...], ...] 长度 K。"""
    cases = sorted(int(c) for c in case_ids)
    if len(cases) < n_folds * 2:
        raise ValueError(
            f"病例数 {len(cases)} 不足以做 {n_folds} 折（每折至少需 2 个测试病例）"
        )
    # 轮转分配，保证各折病例数尽量均衡
    folds = [[] for _ in range(n_folds)]
    for i, c in enumerate(cases):
        folds[i % n_folds].append(c)
    return folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="CT-MRI")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    tag = args.modality.lower().replace("-", "_")
    name = args.name or f"{tag}_cv{args.folds}"
    manifest = os.path.join(HERE, f"manifest_{tag}.csv")

    if not os.path.exists(manifest):
        raise FileNotFoundError(f"缺少清单 {manifest}，请先运行 build_manifest.py 并确认数据存在")

    import csv
    by_case = {}
    with open(manifest, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_case.setdefault(int(row["case_id"]), []).append(row["pair_id"])

    case_ids = sorted(by_case)
    folds = make_folds(case_ids, args.folds)
    print(f"病例总数 {len(case_ids)}，分 {args.folds} 折：")
    for i, cs in enumerate(folds):
        n_img = sum(len(by_case[c]) for c in cs)
        print(f"  折 {i}: 病例 {cs}  ({n_img} 张)")

    out = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": f"Harvard/AANLIB {args.modality}",
        "n_folds": args.folds,
        "n_cases": len(case_ids),
        "n_pairs": sum(len(v) for v in by_case.values()),
        "policy": (
            "按病例号排序后轮转分配到各折，确定性，无随机。"
            "每折：该折病例为 test，从其余病例中取 1 个为 val，其余为 train。"
            "验证病例同样按确定性规则选取（余下病例中编号最小者）。"
        ),
        "warning": (
            "跨折聚合时，独立单元是**病例**而非图像。同一折内 20 张相邻脑切片"
            "高度相关，不可当作 20 个独立样本。"
        ),
        "folds": [],
    }

    for i, test_cases in enumerate(folds):
        rest = [c for c in case_ids if c not in test_cases]
        val_case = rest[0]                      # 确定性：编号最小者
        train_cases = rest[1:]
        fold = {
            "fold": i,
            "test_cases": test_cases,
            "val_cases": [val_case],
            "train_cases": train_cases,
            "splits": {
                "test": sorted([p for c in test_cases for p in by_case[c]], key=int),
                "val": sorted(by_case[val_case], key=int),
                "train": sorted([p for c in train_cases for p in by_case[c]], key=int),
            },
        }
        out["folds"].append(fold)
        print(f"    折 {i}: test={test_cases}  val={[val_case]}  "
              f"train={len(fold['splits']['train'])} 对")

    assert isinstance(out["folds"][0]["splits"]["train"][0], str)

    # 同时给出与 run_experiment.py 兼容的顶层结构（折 0），便于单跑
    out["splits"] = out["folds"][0]["splits"]
    out["split_meta"] = {
        "policy": "折 0（顶层 splits 仅为兼容单次运行；正式的跨折结果见 folds）",
        "train_cases": out["folds"][0]["train_cases"],
        "val_cases": out["folds"][0]["val_cases"],
        "test_cases": out["folds"][0]["test_cases"],
    }

    p = os.path.join(ROOT, "splits", f"{name}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {p}")


if __name__ == "__main__":
    main()
