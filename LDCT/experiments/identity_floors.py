"""
逐患者的 identity 地板（输出=输入），覆盖两个划分的 val + test 患者。

为什么要它
------------------------------------------------------------------
论文 §3.4 的结论是"患者间跨度（23.99–30.33 dB）大于我们报告的任何一个效应，
故一律逐患者报告"。但那张表只列了 L506 / L067 / L143，**没有 val 患者**。

而 val 患者正是训练时唯一能看到的信号：RED-CNN 这类从零训练的网在早期
会低于 identity 地板，只有知道地板在哪，才能判断 21.9 dB（epoch 5）是
"正常的早期状态"还是"训练有问题"。

同时，这张表本身就是论文 §3.4 需要补的内容。

纯 CPU 运行，**不占用 GPU**——可以和 GPU 上的训练并行。

用法
------------------------------------------------------------------
    python experiments/identity_floors.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from data.dataset import AapmDataset      # noqa: E402
import train as T                          # noqa: E402

AAPM_ROOT = os.path.join(ROOT, "data", "aapm_h5")
SPLITS = {
    "S1 (aapm_mayo_3mm)": os.path.join(ROOT, "splits", "aapm_mayo_3mm.json"),
    "S2 (aapm_mayo_3mm_lit)": os.path.join(ROOT, "splits", "aapm_mayo_3mm_lit.json"),
}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    dev = torch.device("cpu")
    out = {}

    for split_name, split_file in SPLITS.items():
        print("=" * 78, flush=True)
        print(f"{split_name}", flush=True)
        print("=" * 78, flush=True)

        for part in ("val", "test"):
            ds = AapmDataset(AAPM_ROOT, split_file, part, cache=True)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            rows, mean = T.evaluate(None, dl, dev, limit=None)

            buckets = defaultdict(list)
            for row, (pid, _) in zip(rows, ds._index):
                buckets[pid].append(row)

            print(f"\n  [{part}] 共 {ds.n_slices} 张，{len(buckets)} 个患者",
                  flush=True)
            print(f"  {'患者':<8}{'n':>6}{'PSNR':>10}{'SSIM':>10}", flush=True)
            for pid in sorted(buckets):
                rs = buckets[pid]
                p = sum(r["PSNR"] for r in rs) / len(rs)
                s = sum(r["SSIM"] for r in rs) / len(rs)
                print(f"  {pid:<8}{len(rs):>6}{p:>10.4f}{s:>10.4f}", flush=True)
                out.setdefault(split_name, {}).setdefault(part, {})[pid] = {
                    "n": len(rs), "PSNR": p, "SSIM": s}

            print(f"  {'合并':<8}{len(rows):>6}{mean['PSNR']:>10.4f}"
                  f"{mean['SSIM']:>10.4f}", flush=True)
            out.setdefault(split_name, {}).setdefault(part, {})["__combined__"] = {
                "n": len(rows), "PSNR": mean["PSNR"], "SSIM": mean["SSIM"]}
        print(flush=True)

    # ---------------------------------------------------------------- 全患者
    # 论文 §3.4 正文写 "The spread across patients (23.99 – 30.33 dB)"，
    # 但表里只列了 4 行、最高 29.2489 —— **30.33 这个数在表里没有出处**。
    # 审稿人会要这个来源。这里把 10 个患者全部算一遍，把跨度坐实。
    print("=" * 78, flush=True)
    print("全部 10 个患者（含训练患者）—— 用于坐实 §3.4 的\"跨度\"一句", flush=True)
    print("=" * 78, flush=True)

    with open(os.path.join(AAPM_ROOT, "index.json"), encoding="utf-8") as f:
        idx = json.load(f)
    all_pats = sorted(idx["patients"])

    tmp_split = os.path.join(HERE, "_all_patients_split.json")
    with open(tmp_split, "w", encoding="utf-8") as f:
        json.dump({"train_patients": [], "val_patients": all_pats,
                   "test_patients": []}, f, ensure_ascii=False)

    ds = AapmDataset(AAPM_ROOT, tmp_split, "val", cache=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows, _ = T.evaluate(None, dl, dev, limit=None)

    buckets = defaultdict(list)
    for row, (pid, _) in zip(rows, ds._index):
        buckets[pid].append(row)

    print(f"\n  {'患者':<8}{'n':>6}{'PSNR':>10}{'SSIM':>10}", flush=True)
    vals = []
    for pid in sorted(buckets):
        rs = buckets[pid]
        p = sum(r["PSNR"] for r in rs) / len(rs)
        s = sum(r["SSIM"] for r in rs) / len(rs)
        vals.append(p)
        print(f"  {pid:<8}{len(rs):>6}{p:>10.4f}{s:>10.4f}", flush=True)
        out.setdefault("all_patients", {})[pid] = {
            "n": len(rs), "PSNR": p, "SSIM": s}

    # ---------------------------------------------------------------- 防呆
    OUT_PATH = os.path.join(HERE, "identity_floors.json")
    # ⚠️ 2026-09-16 踩过的坑：`identity_floors.json` 是**论文 §4.4 明确引用**的产物，
    #    而重下数据后它的切片数会变（435 -> 421）。直接重跑会**静默覆盖**论文引用的
    #    数字，导致正文与产物对不上。
    #    故：若新算出的患者切片数与已有文件不一致，**改写到 _replication 后缀**，
    #    并把这件事打印出来，绝不静默覆盖。
    _n_now = sum(len(v) for v in buckets.values())
    if os.path.exists(OUT_PATH):
        try:
            with open(OUT_PATH, encoding="utf-8") as f:
                _old = json.load(f)
            _n_old = sum(v["n"] for k, v in _old.get("all_patients", {}).items()
                         if k != "__span__")
        except Exception:
            _n_old = None
        if _n_old is not None and _n_old != _n_now:
            alt = OUT_PATH.replace(".json", "_replication.json")
            print(f"\n  ⚠️ 切片数变了（已有 {_n_old} -> 本次 {_n_now}）："
                  f"为避免覆盖论文 §4.4 引用的产物，改写 -> {os.path.basename(alt)}",
                  flush=True)
            OUT_PATH = alt

    lo, hi = min(vals), max(vals)
    print(f"\n  跨度：{lo:.4f} – {hi:.4f} dB", flush=True)
    # 论文当前（§4.4）写的是 "25.50 – 29.25 dB"。
    # ⚠️ 本脚本早先比较的是 "23.99 – 30.33"，那是**更早一版的论文数字**，
    #    与现在的表对不上（表里从来只有 10 行、最高 29.25）。已在论文中更正为
    #    25.50–29.25，这里同步。容差 0.05 dB 足以覆盖缺片带来的微小位移。
    PAPER_LO, PAPER_HI = 25.50, 29.25
    ok_span = abs(lo - PAPER_LO) < 0.05 and abs(hi - PAPER_HI) < 0.05
    print(f"  论文 §4.4 声称 {PAPER_LO:.2f} – {PAPER_HI:.2f} —— "
          f"{'✅ 吻合' if ok_span else '⚠️ 需核对'}", flush=True)
    print(f"  （本次数据 {sum(len(v) for v in buckets.values())} 片；"
          f"原始获取为 435 片——缺片会让个别患者均值轻微位移）", flush=True)
    out["all_patients"]["__span__"] = {"lo": lo, "hi": hi}

    dst = OUT_PATH
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {dst}", flush=True)
    print("\n判读：任何模型在任何患者上低于该患者的 identity 值，\n"
          "      都说明它连'什么都不做'都不如。", flush=True)


if __name__ == "__main__":
    main()
