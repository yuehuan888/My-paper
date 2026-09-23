"""从 span 分片里解出 3mm B30 DICOM。

为什么不复用 download_aapm.sh 里的解包逻辑
================================================================
那版用 **链式走法**（按 `csize` 推到下一个 header）。但本归档在每个条目
之间有 **~320-460 KB 的全零填充**，链式走法一进填充区就断，实测只解出 7 个
文件（应 4756 个）。

正确做法：**扫描 `PK\\x03\\x04` 标记**逐个定位并解析。
（这也是 _verify_span.py 初版误判"链路断裂=数据损坏"的同一个根因。）

注意
================================================================
- 归档用 **ZIP64**：local header 的 csize/usize 是 0xFFFFFFFF，真实值在
  extra 字段的 0x0001 记录里。不处理会得到天文数字并整段跳过。
- 解压后按 `full_3mm/<患者>/full_3mm/*.IMA` 与
  `quarter_3mm/<患者>/quarter_3mm/*.IMA` 落盘，与 prepare_aapm.py 期望一致。

用法
================================================================
    python experiments/extract_aapm_span.py data/raw/aapm_3mmB30_span.bin
"""

from __future__ import annotations

import os
import re
import struct
import sys
import zlib

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
    span = sys.argv[1] if len(sys.argv) > 1 else "data/raw/aapm_3mmB30_span.bin"
    outroot = sys.argv[2] if len(sys.argv) > 2 else "data/raw/aapm_mayo_3mm"

    data = open(span, "rb").read()
    n = len(data)
    print("=" * 74)
    print(f"解包 {span}  ({n/1e9:.2f} GB)")
    print("=" * 74)

    ok = trunc = bad = skip = 0
    counts = {}

    pos = 0
    while True:
        i = data.find(b"PK\x03\x04", pos)
        if i < 0 or i + 30 > n:
            break
        try:
            (_v, fl, method, _t, _d, _crc, csize, usize,
             nlen, elen) = struct.unpack("<HHHHHIIIHH", data[i + 4:i + 30])
            if nlen > 4096:
                pos = i + 4
                skip += 1
                continue
            name = data[i + 30:i + 30 + nlen].decode("utf-8", "replace")
            extra = data[i + 30 + nlen:i + 30 + nlen + elen]
            csize, usize = zip64_fix(extra, csize, usize)
            if csize == U32MAX or usize == U32MAX:
                pos = i + 4
                skip += 1
                continue
            dstart = i + 30 + nlen + elen
            dend = dstart + csize
            if dend > n:
                trunc += 1
                pos = i + 4
                continue
            blob = data[dstart:dend]
            if method == 8:
                try:
                    raw = zlib.decompress(blob, -15)
                except Exception:
                    bad += 1
                    pos = i + 4
                    continue
            elif method == 0:
                raw = blob
            else:
                skip += 1
                pos = i + 4
                continue

            # 只留 3mm B30 的两侧
            m = re.search(r"3mm B30/(full_3mm|quarter_3mm)/\1/(L\d{3})/\1/", name)
            if not m:
                pos = dend
                continue
            dose, pid = m.group(1), m.group(2)
            fn = os.path.basename(name)
            dst = os.path.join(outroot, dose, pid, dose, fn)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(raw)
            ok += 1
            counts[(dose, pid)] = counts.get((dose, pid), 0) + 1
            pos = dend
        except Exception:
            pos = i + 4

    print(f"\n解出: {ok}   截断跳过: {trunc}   解压失败: {bad}   其他跳过: {skip}")
    print("\n逐患者统计:")
    pats = sorted({p for (_d, p) in counts})
    for p in pats:
        print(f"   {p}: full={counts.get(('full_3mm', p),0):4d}  "
              f"quarter={counts.get(('quarter_3mm', p),0):4d}")
    print(f"\n输出目录: {os.path.abspath(outroot)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
