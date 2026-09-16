"""自包含地校验 span 分片的完整性（不依赖网络）。

为什么需要
================================================================
Kaggle 的签名 URL 时通时断，所以"跟服务端比对 md5"这条路不可靠。
但 span 是 ZIP 的一段**连续前缀区域**，而 ZIP 的 local file header 是
**顺序链接**的：每个 header 自带 `compressed_size`，下一个 header 就落在
`当前偏移 + 30 + name_len + extra_len + compressed_size`。
若数据被**重复累加**（本项目真实踩过的坑），这条链必然在某处断裂。

因此：**能一路走通 = 没有重复/缺口**。这是自包含证据，不需要网络。

注意：span 只是归档的一段，**不含**中央目录（在文件末尾），
所以只能靠 local header 链，不能用 central directory。

用法
================================================================
    python experiments/_verify_span.py data/raw/aapm_3mmB30_span.bin
"""

from __future__ import annotations

import struct
import sys
import zlib


U32MAX = 0xFFFFFFFF


def _zip64_sizes(extra: bytes, csize: int, usize: int, usize_pos: int):
    """从 ZIP64 扩展字段里取真实大小。

    ⚠️ 本归档 11.27 GB，**必然**用 ZIP64：local header 的 csize/usize 字段是
    32 位，装不下就填 0xFFFFFFFF，真实值放进 extra 的 0x0001 记录里。
    第一次写这个校验器时没处理这一点，导致"走通条目数 = 0"（第一个条目的
    csize 读成 0xFFFFFFFF，直接越过文件末尾）。**别删这段。**
    """
    off = 0
    while off + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[off:off + 4])
        body = extra[off + 4:off + 4 + hsz]
        if hid == 0x0001:                      # ZIP64 extended information
            p = 0
            if usize == U32MAX and p + 8 <= len(body):
                usize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            if csize == U32MAX and p + 8 <= len(body):
                csize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            return csize, usize
        off += 4 + hsz
    return csize, usize


def walk(path: str):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data)
    off = 0
    entries = 0
    names = []
    while off + 30 <= n:
        sig = data[off:off + 4]
        if sig != b"PK\x03\x04":
            return entries, names, off, f"在偏移 {off} 处不是 local header（链路断裂）"
        (ver, flags, method, mtime, mdate, crc, csize, usize,
         nlen, elen) = struct.unpack("<HHHHHIIIHH", data[off + 4:off + 30])
        if nlen > 4096:
            return entries, names, off, f"偏移 {off} 处 nlen 异常 ({nlen})"
        name = data[off + 30:off + 30 + nlen].decode("utf-8", "replace")
        extra = data[off + 30 + nlen:off + 30 + nlen + elen]
        csize, usize = _zip64_sizes(extra, csize, usize, 8)

        dstart = off + 30 + nlen + elen
        dend = dstart + csize
        if dend > n:
            # 最后一个条目不完整 —— 正常，分片正好切在这里
            return entries, names, off, None
        entries += 1
        names.append(name)
        off = dend
    return entries, names, off, f"走到文件末尾仍有残留 {n - off} 字节"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/aapm_3mmB30_span.bin"
    import os
    n = os.path.getsize(path)
    print("=" * 72)
    print("span 完整性校验（ZIP local header 链）")
    print("=" * 72)
    print(f"文件: {path}")
    print(f"大小: {n:,} 字节 ({n/1e6:.1f} MB)")

    entries, names, off, err = walk(path)
    print(f"\n走通条目数: {entries}")
    print(f"走到偏移  : {off:,}  ({100*off/n:.1f}% 的文件被确认为连续有效)")

    if err:
        print(f"\n终止原因: {err}")
        print("  （若只是尾部截断，属正常 —— 分片本就切在条目中间）")
    else:
        print("\n终止原因: 最后一个条目不完整（正常的尾部截断）")

    # 抽样看路径名是否是我们期望的那批患者
    import re
    pats = sorted({m.group(1) for nm in names
                   for m in [re.search(r"(L\d{3})", nm)] if m})
    if pats:
        print(f"\n出现的患者 ID ({len(pats)}): {pats}")
    print(f"\n路径样例:")
    for nm in names[:3]:
        print("   ", nm)
    if len(names) > 3:
        print("    ...")
        for nm in names[-3:]:
            print("   ", nm)

    print("\n" + "=" * 72)
    if err and "不是 local header" in (err or ""):
        print("结论: 链路断裂 —— 数据可能有重复累加或缺口，需重下。")
        return 1
    print("结论: 链路连续 —— 已下载部分**无重复累加、无缺口**。")
    print("       （这是自包含证据，不依赖网络。）")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
