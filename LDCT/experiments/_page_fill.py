"""逐页油墨覆盖率 —— 找出"几乎空白"的页。

为什么需要：float 密集的文档里 LaTeX 会生成只放一两个浮动体的页，
看起来像排版事故。肉眼看几张页会漏，逐页量一遍才有全局判断。
覆盖率 = 非白像素占比。正文页通常 0.10-0.20，纯浮动页会明显偏低。

用法
    python experiments/_page_fill.py <pdf> [--render 目录]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pymupdf


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    pdf = sys.argv[1] if len(sys.argv) > 1 else "paper/latex/main.pdf"
    outdir = None
    if "--render" in sys.argv:
        outdir = sys.argv[sys.argv.index("--render") + 1]
        os.makedirs(outdir, exist_ok=True)

    doc = pymupdf.open(pdf)
    print("=" * 62)
    print("逐页油墨覆盖率: %s  (%d 页)" % (pdf, doc.page_count))
    print("=" * 62)

    sparse = []
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=60, colorspace=pymupdf.csGRAY)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        ink = float((a < 245).mean())
        bar = "#" * int(ink * 100)
        flag = ""
        if ink < 0.06:
            flag = "   <== 近乎空白"
            sparse.append(i + 1)
        print("  p%-3d %.3f  %s%s" % (i + 1, ink, bar[:40], flag))
        if outdir:
            page.get_pixmap(dpi=100).save(os.path.join(outdir, "p%02d.png" % (i + 1)))

    print()
    mean = np.mean([1 for _ in range(1)])  # placeholder to keep output tidy
    if sparse:
        print("近乎空白的页: %s" % sparse)
    else:
        print("无近乎空白的页。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
