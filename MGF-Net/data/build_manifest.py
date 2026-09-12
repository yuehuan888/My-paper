"""
从 Harvard/AANLIB 数据集构建可追溯的清单与**病例级**划分。

背景
------------------------------------------------------------------
`xianming-gu/ASFE-Fusion` 发布的 CT-MRI 子集共 184 对图，被 ECFusion、
ASFE-Fusion、EH-DRAN、TTTFusion 等多篇论文使用。但 ID 不是随机的：

    id // 1000 = 病例号      id % 1000 = 切层号

实测 CT-MRI 的 184 张 = **10 个病例 × 16–21 层**（CT 与 MRI 病例号完全一致）。

因此"从 184 中随机选 24 对做测试集"这个领域惯例存在**数据泄漏**：
同一病例的相邻切片高度相似，随机划分会把它们同时放进训练集与测试集。

本脚本据此建立：
  1. `data/manifest_<modality>.csv` —— 逐图清单（病例号、切层号、路径、哈希、尺寸）
  2. `splits/<name>.json` —— **按病例**划分的 train/val/test，并附上
     "切片级随机划分"作为对照，便于量化泄漏带来的虚高

用法
------------------------------------------------------------------
    python data/build_manifest.py --modality CT-MRI --name ct_mri_case_v1

输出划分规则（10 例 → 7/1/2）：
    病例号排序后，前 7 个为 train，第 8 个为 val，后 2 个为 test。
    确定性分配，无随机，任何人可复现。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATASET_MARKER = "Havard-Medical-Image-Fusion-Datasets-main"

# 每个模态的 (输入A 子目录, MRI 子目录)
SUBDIRS = {
    "CT-MRI": ("CT", "MRI"),
    "PET-MRI": ("PET", "MRI"),
    "SPECT-MRI": ("SPECT", "MRI"),
}


def find_dataset_root():
    for base in [os.path.join(HERE, "harvard_full"), HERE]:
        for dirpath, dirnames, _ in os.walk(base):
            if DATASET_MARKER in dirnames:
                return os.path.join(dirpath, DATASET_MARKER)
            if os.path.basename(dirpath) == DATASET_MARKER:
                return dirpath
    raise FileNotFoundError(
        f"未找到数据集目录 {DATASET_MARKER}。"
        "请先解压 hmif.tar.gz 到 MGF-Net/data/harvard_full/"
    )


def parse_id(img_id: str):
    """id // 1000 = 病例号，id % 1000 = 切层号。"""
    n = int(img_id)
    return n // 1000, n % 1000


def file_sha256(path: str, limit: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(limit))
    return h.hexdigest()[:16]


def build(modality: str, out_csv: str):
    root = find_dataset_root()
    sub_a, sub_mri = SUBDIRS[modality]
    dir_a = os.path.join(root, modality, sub_a)
    dir_mri = os.path.join(root, modality, sub_mri)

    if not os.path.isdir(dir_a):
        raise FileNotFoundError(f"{dir_a} 不存在")

    def scan(d):
        return {os.path.splitext(f)[0]: os.path.join(d, f)
                for f in os.listdir(d) if f.lower().endswith(".png")}

    A, B = scan(dir_a), scan(dir_mri)
    common = sorted(set(A) & set(B), key=int)

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    if only_a or only_b:
        print(f"⚠️  {modality} 配对不完整：仅 {sub_a} 有 {len(only_a)}，"
              f"仅 {sub_mri} 有 {len(only_b)}", file=sys.stderr)

    rows = []
    for i in common:
        case, sl = parse_id(i)
        sz_a = Image.open(A[i]).size
        sz_b = Image.open(B[i]).size
        rows.append({
            "pair_id": i,
            "modality": modality,
            "case_id": case,
            "slice_id": sl,
            "path_a": os.path.relpath(A[i], ROOT).replace("\\", "/"),
            "path_b": os.path.relpath(B[i], ROOT).replace("\\", "/"),
            "size_a": f"{sz_a[0]}x{sz_a[1]}",
            "size_b": f"{sz_b[0]}x{sz_b[1]}",
            "sha256_a": file_sha256(A[i]),
            "sha256_b": file_sha256(B[i]),
        })

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    return rows


def make_splits(rows, name: str):
    """按病例划分，并额外给出"切片级随机划分"作为泄漏对照。"""
    by_case = defaultdict(list)
    for r in rows:
        by_case[r["case_id"]].append(r["pair_id"])

    cases = sorted(by_case)
    n = len(cases)
    if n < 5:
        raise ValueError(f"病例数仅 {n}，不足以做病例级划分")

    # 7 : 1 : 2 的确定性分配（按病例号排序后切分）
    n_test = max(2, round(n * 0.2))
    n_val = max(1, round(n * 0.1))
    n_train = n - n_val - n_test
    tr_cases = cases[:n_train]
    va_cases = cases[n_train:n_train + n_val]
    te_cases = cases[n_train + n_val:]

    def ids_of(cs):
        return sorted([i for c in cs for i in by_case[c]], key=int)

    case_split = {
        "train": ids_of(tr_cases), "val": ids_of(va_cases), "test": ids_of(te_cases),
    }

    # 泄漏对照：在所有 pair 上做切片级随机划分，保持相同的测试集大小
    rng = np.random.default_rng(0)
    all_ids = sorted([r["pair_id"] for r in rows], key=int)
    perm = rng.permutation(len(all_ids))
    n_te = len(case_split["test"])
    n_va = len(case_split["val"])
    slice_split = {
        "test": sorted([all_ids[k] for k in perm[:n_te]], key=int),
        "val": sorted([all_ids[k] for k in perm[n_te:n_te + n_va]], key=int),
        "train": sorted([all_ids[k] for k in perm[n_te + n_va:]], key=int),
    }

    def cases_in(idlist):
        return sorted({int(i) // 1000 for i in idlist})

    out = {
        "name": name,
        "created": datetime.now(timezone.utc).isoformat(),
        "dataset": "Harvard/AANLIB CT-MRI，经 xianming-gu/ASFE-Fusion 发布",
        "n_pairs": len(rows),
        "n_cases": len(cases),
        "id_convention": "id // 1000 = 病例号；id % 1000 = 切层号（实测于本数据集）",
        "warning": (
            "本划分为病例级，目的是避免同一病例相邻切片同时进入训练与测试集。"
            "领域惯例是在切片级随机划分，存在数据泄漏，见 slice_level_random 字段。"
        ),
        "splits": case_split,
        "split_meta": {
            "policy": "按病例号排序后 7:1:2 确定性分配，无随机",
            "train_cases": tr_cases,
            "val_cases": va_cases,
            "test_cases": te_cases,
        },
        "slice_level_random": {
            "note": "对照用：领域惯例的切片级随机划分（seed=0），用于量化泄漏带来的虚高",
            "splits": slice_split,
            "meta": {
                "train_cases": cases_in(slice_split["train"]),
                "val_cases": cases_in(slice_split["val"]),
                "test_cases": cases_in(slice_split["test"]),
            },
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", default="CT-MRI", choices=list(SUBDIRS))
    ap.add_argument("--name", default="ct_mri_case_v1")
    args = ap.parse_args()

    tag = args.modality.lower().replace("-", "_")
    csv_path = os.path.join(HERE, f"manifest_{tag}.csv")
    rows = build(args.modality, csv_path)

    split = make_splits(rows, args.name)
    split_path = os.path.join(ROOT, "splits", f"{args.name}.json")
    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"清单      : {csv_path}  ({len(rows)} 对)")
    print(f"划分      : {split_path}")
    print()
    print(f"病例总数  : {split['n_cases']}")
    m = split["split_meta"]
    print(f"  训练病例: {m['train_cases']}")
    print(f"  验证病例: {m['val_cases']}")
    print(f"  测试病例: {m['test_cases']}")
    print()
    for k in ["train", "val", "test"]:
        print(f"  {k:<6}: {len(split['splits'][k]):>3} 对")
    print()
    sl = split["slice_level_random"]
    print("泄漏对照（切片级随机划分）各集合覆盖的病例：")
    print(f"  训练: {sl['meta']['train_cases']}")
    print(f"  测试: {sl['meta']['test_cases']}")
    overlap = set(sl["meta"]["train_cases"]) & set(sl["meta"]["test_cases"])
    print(f"  ⚠️ 训练与测试共享病例: {sorted(overlap)}  —— 这就是泄漏")


if __name__ == "__main__":
    main()
