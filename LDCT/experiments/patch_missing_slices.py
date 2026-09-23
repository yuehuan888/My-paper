"""定点补齐 span 缺失的 DICOM 切片。

背景
================================================================
span 分片是"归档中一段连续字节"。实测发现该区间内**有洞**：
L067 全剂量的 24 号损坏、25-36 号整段不存在（12 张缺失），
尽管 span 的总长度恰好等于预期（1,481,701,181 字节）。
即"3mm B30 在归档内完全连续"这个假设**不成立**。

好在 `ishak21/aapm-cst` 保留了完整全剂量集（2378 张，10 患者）。
**不必下整包（0.85 GB）**：先读它的中央目录定位目标条目，
再对每个条目做一次小 Range 请求，13 张 × ~330 KB ≈ 4 MB。

用法
================================================================
    python experiments/patch_missing_slices.py --list
    python experiments/patch_missing_slices.py --fetch
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import struct
import subprocess
import sys
import time
import zlib

REPO = "ishak21/aapm-cst"
# 需要补的：L067 全剂量 24-36 号（24 号损坏，25-36 缺失）
NEED = {"L067": list(range(24, 37))}
OUTROOT = "data/raw/aapm_mayo_3mm"

# ⚠️ 这里**不能**带 `-L`：取签名 URL 靠的是 `-w "%{redirect_url}"`，
#    而 `-L` 会让 curl 一路跟着重定向，把整个归档下到 /dev/null 才返回 ——
#    既慢又必然超时（实测表现为"取不到签名 URL"）。取 URL 时必须停在 302。
CURL_NOL = ["curl", "-s", "--ssl-revoke-best-effort"]
CURL = ["curl", "-sL", "--ssl-revoke-best-effort"]


def signed_url(retries: int = 15) -> str:
    for _ in range(retries):
        p = subprocess.run(CURL_NOL + ["-o", os.devnull, "-w", "%{redirect_url}",
                                       "--max-time", "25",
                                       f"https://www.kaggle.com/api/v1/datasets/download/{REPO}"],
                           capture_output=True, timeout=60)
        u = p.stdout.decode("utf-8", "replace").strip()
        if u:
            return u
        time.sleep(3)
    return ""


def fetch_range(url: str, start: int, end: int, retries: int = 4):
    for _ in range(retries):
        p = subprocess.run(CURL + ["--max-time", "90", "-r", f"{start}-{end}", url],
                           capture_output=True, timeout=150)
        if p.returncode == 0 and p.stdout:
            return p.stdout
        time.sleep(2)
    return None


def read_central_dir(url: str):
    """读 ZIP 尾部取中央目录，返回 [(name, local_header_offset, csize, usize), ...]。"""
    # ⚠️ 不能用 HEAD 取大小：Kaggle 的 HEAD 返回 404（实测；agent 也独立发现过）。
    #    改用 GET + Range 读 `Content-Range: bytes 0-0/<TOTAL>`。
    size = None
    p = subprocess.run(CURL + ["-D", "-", "-o", os.devnull, "--max-time", "60",
                               "-r", "0-0", url], capture_output=True, timeout=120)
    for line in p.stdout.decode("utf-8", "replace").split("\n"):
        if line.lower().startswith("content-range:"):
            try:
                size = int(line.split("/")[1].strip())
            except Exception:
                pass
    if not size:
        print("  !! 无法取得归档大小")
        return []
    print(f"  归档大小: {size:,}")

    tail_len = min(8 << 20, size)
    tail = fetch_range(url, size - tail_len, size - 1)
    if not tail:
        print("  !! 取尾部失败")
        return []
    entries = []
    pos = 0
    while True:
        j = tail.find(b"PK\x01\x02", pos)
        if j < 0:
            break
        try:
            (csize, usize, nlen, elen, clen) = struct.unpack("<IIIHH", tail[j + 20:j + 36])
            off = struct.unpack("<I", tail[j + 42:j + 46])[0]
            name = tail[j + 46:j + 46 + nlen].decode("utf-8", "replace")
            extra = tail[j + 46 + nlen:j + 46 + nlen + elen]
            # ZIP64：真实 offset 在 extra 的 0x0001 里
            if off == 0xFFFFFFFF:
                k = 0
                while k + 4 <= len(extra):
                    hid, hsz = struct.unpack("<HH", extra[k:k + 4])
                    body = extra[k + 4:k + 4 + hsz]
                    if hid == 0x0001:
                        q = 0
                        if usize == 0xFFFFFFFF:
                            q += 8
                        if csize == 0xFFFFFFFF:
                            q += 8
                        if off == 0xFFFFFFFF and q + 8 <= len(body):
                            off = struct.unpack("<Q", body[q:q + 8])[0]
                        break
                    k += 4 + hsz
            entries.append((name, off, csize, usize))
        except Exception:
            pass
        pos = j + 4
    return entries


def local_data_offset(url: str, header_off: int):
    """读 local header，返回 (数据起始, csize, usize, method)。"""
    hdr = fetch_range(url, header_off, header_off + 1023)
    if not hdr or hdr[:4] != b"PK\x03\x04":
        return None
    (_v, _fl, method, _t, _d, _crc, csize, usize,
     nlen, elen) = struct.unpack("<HHHHHIIIHH", hdr[4:30])
    extra = hdr[30 + nlen:30 + nlen + elen]
    # ZIP64 修正
    if csize == 0xFFFFFFFF or usize == 0xFFFFFFFF:
        k = 0
        while k + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[k:k + 4])
            body = extra[k + 4:k + 4 + hsz]
            if hid == 0x0001:
                p = 0
                if usize == 0xFFFFFFFF and p + 8 <= len(body):
                    usize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                if csize == 0xFFFFFFFF and p + 8 <= len(body):
                    csize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
                break
            k += 4 + hsz
    return (header_off + 30 + nlen + elen, csize, usize, method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()

    url = signed_url()
    if not url:
        sys.exit("取不到签名 URL")
    print(f"签名 URL 已取得 (len={len(url)})")

    ents = read_central_dir(url)
    print(f"中央目录条目: {len(ents)}")
    if not ents:
        sys.exit(1)

    # 挑出需要的。
    # ⚠️ 必须同时要求 `.IMA` 结尾：每个切片目录下还有 `50/100/150/data.csv`，
    #    它们的路径里同样含 `L067_FD_3_1.CT.0002.0025.`，只按目录名匹配会误命中
    #    （实测第一版就抓成了 data.csv）。
    want = []
    for name, off, cs, us in ents:
        if not name.lower().endswith(".ima"):
            continue
        m = re.search(r"/(L\d{3})/(L\d{3})_(FD|QD)_3_1\.CT\.\d{4}\.(\d{4})\.", name)
        if not m:
            continue
        pid, dose, num = m.group(1), m.group(3), int(m.group(4))
        if dose != "FD":
            continue
        if pid in NEED and num in NEED[pid]:
            want.append((name, off, cs, us, pid, num))
    want.sort(key=lambda t: (t[4], t[5]))

    print(f"\n目标条目: {len(want)}")
    for name, off, cs, us, pid, num in want:
        print(f"   {pid} #{num:>4}  off={off:>12,}  csize={cs:>8,}")

    if args.list or not want:
        return 0

    if args.fetch:
        ok = 0
        for name, off, cs, us, pid, num in want:
            info = local_data_offset(url, off)
            if not info:
                print(f"   [{pid} #{num}] 读 header 失败")
                continue
            dstart, csize, usize, method = info
            blob = fetch_range(url, dstart, dstart + csize - 1)
            if not blob:
                print(f"   [{pid} #{num}] 取数据失败")
                continue
            raw = zlib.decompress(blob, -15) if method == 8 else blob
            if len(raw) != usize:
                print(f"   [{pid} #{num}] 尺寸不符 {len(raw)} != {usize}")
                continue
            fn = os.path.basename(name)
            dst = os.path.join(OUTROOT, "full_3mm", pid, "full_3mm", fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(raw)
            ok += 1
            print(f"   [{pid} #{num}] OK  {usize:,} bytes -> {fn[:40]}...")
        print(f"\n补齐 {ok}/{len(want)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
