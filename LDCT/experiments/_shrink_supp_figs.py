"""把"Supplementary figures"一节里的 5 张图从整栏宽降到 0.82 倍栏宽。

为什么：编译实测这两张页近乎空白（p15 只放 Fig 12、p18 只放 Fig 17），
根因是 28 个跨栏浮动体排队。补充图是这批浮动体里最可压缩的一档 ——
0.82 倍仍有约 5.9 英寸宽，比正文里的单栏图（3.5 英寸）大得多。

只动 Supplementary 一节，不碰正文图。
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "latex", "main.tex")

SUPP_FIGS = ("fig5_slice_dist.png", "fig6_convergence.png", "fig7_forest.png",
             "fig9_perpatient.png", "fig12_effect_forest.png")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    s = io.open(TEX, encoding="utf-8").read()
    marker = "section*{Supplementary figures}"
    if marker not in s:
        print("[FAIL] 找不到 Supplementary figures 一节")
        return 1
    i = s.index(marker)
    head, tail = s[:i], s[i:]

    n = 0
    for name in SUPP_FIGS:
        old = "\\includegraphics[width=\\textwidth]{%s}" % name
        new = "\\includegraphics[width=0.82\\textwidth]{%s}" % name
        if old in tail:
            tail = tail.replace(old, new)
            n += 1
        else:
            print("  [警告] 未找到 %s 的整栏宽写法" % name)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(head + tail)
    print("改写补充图 %d/5 张 -> width=0.82\\textwidth" % n)

    # 回读校验
    chk = io.open(TEX, encoding="utf-8").read()
    still = [x for x in SUPP_FIGS if "width=\\textwidth]{%s}" % x in chk]
    print("回读校验:", "PASS" if not still else "[FAIL] 仍为整栏宽: %s" % still)
    return 0 if not still else 1


if __name__ == "__main__":
    sys.exit(main())
