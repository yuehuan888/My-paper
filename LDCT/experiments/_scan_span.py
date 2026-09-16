"""扫描已下载的 span，列出其中包含哪些患者的哪些文件。

为什么不复用 _verify_span.py 的链式走法：
    该归档在每个条目之间有 **~320-460 KB 的全零填充**，所以"按 csize 前进"
    会立刻踩进填充区报断链。正确的做法是**扫描** `PK\\x03\\x04` 标记，
    再逐个解析（这正是 _verify_span 初版误判为"损坏"的原因）。

用途：在源数据集已被下架的情况下，弄清手上这 765 MB 究竟覆盖了哪些内容。
"""

from __future__ import annotations

import collections
import os
import re
import struct
import sys

U32MAX = 0xFFFFFFFF


def zip64_fix(extra: bytes, csize: int, usize: int):
    off = 0
    while off + 4 <= len(extra):
        hid, hsz = struct.unpack("<HH", extra[off:off + 4])
        body = extra[off + 4:off + 4 + hsz]
        if hid == 0x0001:
            p = 0
            if usize == U32MAX and p + 8 <= len(body):
                usize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            if csize == U32MAX and p + 8 <= len(body):
                csize = struct.unpack("<Q", body[p:p + 8])[0]; p += 8
            return csize, usize
        off += 4 + hsz
    return csize, usize


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/aapm_3mmB30_span.bin"
    data = open(path, "rb").read()
    n = len(data)
    print("=" * 74)
    print(f"扫描: {path}   {n:,} 字节 ({n/1e6:.1f} MB)")
    print("=" * 74)

    names = []
    pos = 0
    while True:
        i = data.find(b"PK\x03\x04", pos)
        if i < 0 or i + 30 > n:
            break
        try:
            (_ver, _fl, _m, _t, _d, _crc, csize, usize,
             nlen, elen) = struct.unpack("<HHHHHIIIHH", data[i + 4:i + 30])
            if nlen > 4096:
                pos = i + 4
                continue
            nm = data[i + 30:i + 30 + nlen].decode("utf-8", "replace")
            extra = data[i + 30 + nlen:i + 30 + nlen + elen]
            csize, usize = zip64_fix(extra, csize, usize)
            if csize == U32MAX or usize == U32MAX:
                pos = i + 4
                continue
            names.append((i, nm, csize))
        except Exception:
            pass
        pos = i + 4

    print(f"\n识别到条目: {len(names)}")
    if not names:
        return 1

    # 哪些条目是**完整**的（数据段没被文件尾部截断）
    complete = [(o, nm, cs) for (o, nm, cs) in names if o + 30 + len(nm) + cs <= n]
    print(f"其中完整（数据段未越界）: {len(complete)}")
    print(f"最后一条完整条目结束于偏移: "
          f"{max((o + 30 + len(nm) + cs) for o, nm, cs in complete):,}"
          f"  ({100*max((o+30+len(nm)+cs) for o, nm, cs in complete)/n:.1f}% 的文件)")

    # 按患者 / 剂量 统计
    print("\n按 患者-剂量 统计（仅完整条目）:")
    pat = re.compile(r"(L\d{3})_(FD|QD|ND)")
    cnt = collections.Counter()
    for _o, nm, _cs in complete:
        m = pat.search(nm)
        cnt[m.group(0) if m else "?"] += 1
    for k in sorted(cnt):
        print(f"   {k}: {cnt[k]}")

    print("\n前 5 个条目:")
    for o, nm, cs in names[:5]:
        print(f"   [{o:>12,}] {cs:>9,}B  {nm[-60:]}")
    print("\n后 5 个条目:")
    for o, nm, cs in names[-5:]:
        print(f"   [{o:>12,}] {cs:>9,}B  {nm[-60:]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
