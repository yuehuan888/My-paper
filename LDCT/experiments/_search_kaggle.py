"""在 Kaggle 上系统搜索 AAPM-Mayo 低剂量 CT 数据集的可用镜像。

为什么要搜：论文需要 **成对** 的 3mm B30（输入=quarter_3mm，真值=full_3mm）。
原源 `abhishekpjijiju/training-image-data` 已转私有/删除（页面 404、API 403）。
已确认 `ishak21/aapm-cst` 只有全剂量（无 QD），配不成对。

判据：候选必须**含低剂量**那一半（文件名 `_QD_` 或路径 `quarter_3mm`）。

用法：
    python experiments/_search_kaggle.py
    python experiments/_search_kaggle.py --inspect tridibjyotidas/aapm-ct-1mmb30
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
import time

CURL = ["curl", "-sL", "--ssl-revoke-best-effort", "--max-time", "40"]
API = "https://www.kaggle.com/api/v1"

# 覆盖多种命名习惯：数据集作者对同一份数据的命名差异很大
QUERIES = [
    "aapm", "mayo", "low dose ct", "ldct", "quarter dose",
    "aapm 2016", "grand challenge", "ct denoising", "low-dose",
    "aapm-mayo", "mayo clinic ct", "ldct denoising", "sinogram",
    "ct reconstruction", "medical ct", "3mm",
]


def get(url: str, retries: int = 4):
    for i in range(retries):
        try:
            p = subprocess.run(CURL + [url], capture_output=True, timeout=60)
            if p.returncode == 0 and p.stdout and len(p.stdout) > 2:
                return p.stdout
        except Exception:
            pass
        time.sleep(2)
    return None


def search():
    seen = {}
    for q in QUERIES:
        raw = get(f"{API}/datasets/list?search={q.replace(' ', '+')}&page=1")
        if not raw:
            print(f"  [{q}] 请求失败")
            continue
        try:
            ds = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if not isinstance(ds, list):
            ds = ds.get("datasets", [])
        for x in ds:
            ref = x.get("ref") or f"{x.get('ownerName')}/{x.get('datasetSlug')}"
            seen[ref] = {
                "title": (x.get("title") or "")[:60],
                "size": x.get("totalBytes") or 0,
                "queries": seen.get(ref, {}).get("queries", []) + [q],
            }
        time.sleep(0.5)

    print(f"\n共 {len(seen)} 个去重数据集。按大小排序，标注命中的关键词：\n")
    rows = sorted(seen.items(), key=lambda kv: -kv[1]["size"])
    for ref, m in rows:
        gb = m["size"] / 1e9
        if gb < 0.05:
            continue
        print(f"  {ref:<58} {gb:6.2f} GB  {m['title'][:42]}")
    return seen


def inspect(ref: str):
    """读 ZIP 中央目录，判断是否含 QD。"""
    print(f"\n=== 检查 {ref} ===")
    u = get(f"{API}/datasets/download/{ref}")
    # 需要拿到 redirect_url，用 -w 输出
    p = subprocess.run(CURL + ["-o", os.devnull, "-w", "%{redirect_url}",
                               f"{API}/datasets/download/{ref}"],
                       capture_output=True, timeout=60)
    url = p.stdout.decode("utf-8", "replace").strip()
    if not url:
        print("  取不到签名 URL")
        return
    # 取尾部 4MB 读中央目录
    tmp = r"D:\huan.yue\mypaper\My-paper\LDCT\data\_probe\insp.bin"
    p2 = subprocess.run(CURL + ["--max-time", "120", "-r", "-4194304",
                                "-o", tmp, url], capture_output=True, timeout=180)
    import struct
    import collections
    import re
    try:
        d = open(tmp, "rb").read()
    except Exception as e:
        print("  取样失败:", e)
        return
    names = []
    pos = 0
    while True:
        j = d.find(b"PK\x01\x02", pos)
        if j < 0:
            break
        try:
            nlen = struct.unpack("<H", d[j + 28:j + 30])[0]
            names.append(d[j + 46:j + 46 + nlen].decode("utf-8", "replace"))
        except Exception:
            pass
        pos = j + 4
    print(f"  可见中央目录条目: {len(names)}")
    if not names:
        return
    ext = collections.Counter(n.rsplit(".", 1)[-1].lower() for n in names if "." in n)
    print(f"  扩展名分布: {dict(ext.most_common(6))}")
    dose = collections.Counter()
    for n in names:
        m = re.search(r"_(FD|QD|ND)_", n)
        if m:
            dose[m.group(1)] += 1
    print(f"  剂量标记: {dict(dose)}  <-- 必须同时有 FD 和 QD 才可用")
    pats = sorted({m.group(1) for n in names for m in [re.search(r"(L\d{3})", n)] if m})
    print(f"  患者: {len(pats)} 个 {pats[:12]}")
    for n in names[:3]:
        print("   ", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", default=None)
    args = ap.parse_args()
    if args.inspect:
        inspect(args.inspect)
    else:
        search()


if __name__ == "__main__":
    import os
    sys.exit(main())
