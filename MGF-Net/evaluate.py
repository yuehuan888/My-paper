"""
融合结果批量评价工具。

用法：
    python evaluate.py --fused-dir <目录> --ct-dir <目录> --mri-dir <目录> [--out results.csv]

行为：
    - 从 --fused-dir 读取融合图，按**文件名**与 CT/MRI 显式配对（不做排序后 zip）
    - 缺失配对、尺寸不一致、数量不符一律**报错**，不静默截断
    - 输出逐图 CSV，并打印按图聚合的均值

⚠️ 与旧 `compare_baselines.py` 的关键区别：
    - 旧版把 train 与 test 图混在一起出总分，产出的数字不能当测试成绩
    - 旧版用全图统计 SSIM，等价于窗口=整张图的退化实现

    **train / val / test 必须分开目录、分开报告**，本工具只管评价，
    划分由调用方保证。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.metrics import evaluate_pair  # noqa: E402

METRIC_KEYS = ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _list_ids(d):
    if not os.path.isdir(d):
        raise FileNotFoundError(f"目录不存在: {d}")
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.lower().endswith(IMG_EXTS)
    )


def _load(path):
    """读为 uint8。

    注意不要读成 float64 —— 那会得到 [0,255] 域的 float，
    被 metrics.as_255 的值域校验拒绝（这是刻意的自我保护：
    float 只接受 [0,1]，避免与 [0,255] 的 float 产生歧义）。
    """
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def _find(d, img_id):
    for ext in IMG_EXTS:
        p = os.path.join(d, img_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"在 {d} 中找不到 {img_id}（已尝试 {'/'.join(IMG_EXTS)}）")


def evaluate_dirs(fused_dir, ct_dir, mri_dir, out_csv=None, quiet=False):
    fused_ids = _list_ids(fused_dir)
    ct_ids = set(_list_ids(ct_dir))
    mri_ids = set(_list_ids(mri_dir))

    if not fused_ids:
        raise ValueError(f"{fused_dir} 中没有图像")

    # 显式校验，不静默截断
    missing = []
    for i in fused_ids:
        if i not in ct_ids:
            missing.append(f"CT 缺少 {i}")
        if i not in mri_ids:
            missing.append(f"MRI 缺少 {i}")
    if missing:
        raise ValueError("配对失败，缺失如下（不静默跳过）:\n  " + "\n  ".join(missing))

    rows = []
    skipped = []
    for img_id in fused_ids:
        f = _load(_find(fused_dir, img_id))
        a = _load(_find(ct_dir, img_id))
        b = _load(_find(mri_dir, img_id))

        if not (f.shape == a.shape == b.shape):
            skipped.append(f"{img_id}: 尺寸不一致 fused={f.shape} ct={a.shape} mri={b.shape}")
            continue

        m = evaluate_pair(f, a, b)
        row = {"id": img_id}
        row.update({k: m[k] for k in METRIC_KEYS})
        rows.append(row)

    if skipped:
        print("⚠️  以下样本被跳过（尺寸不一致）:", file=sys.stderr)
        for s in skipped:
            print("   " + s, file=sys.stderr)

    if not rows:
        raise ValueError("没有任何样本通过校验")

    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id"] + METRIC_KEYS)
            w.writeheader()
            w.writerows(rows)

    if not quiet:
        print(f"\n{'id':<14}" + "".join(f"{k:>9}" for k in METRIC_KEYS))
        print("-" * (14 + 9 * len(METRIC_KEYS)))
        for r in rows:
            print(f"{r['id']:<14}" + "".join(f"{r[k]:>9.4f}" for k in METRIC_KEYS))
        print("-" * (14 + 9 * len(METRIC_KEYS)))
        mean = {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}
        print(f"{'MEAN(n=' + str(len(rows)) + ')':<14}"
              + "".join(f"{mean[k]:>9.4f}" for k in METRIC_KEYS))
        if out_csv:
            print(f"\n已写入 {out_csv}")

    return rows


def main():
    ap = argparse.ArgumentParser(description="融合结果批量评价")
    ap.add_argument("--fused-dir", required=True)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--mri-dir", required=True)
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    args = ap.parse_args()
    evaluate_dirs(args.fused_dir, args.ct_dir, args.mri_dir, args.out)


if __name__ == "__main__":
    main()
