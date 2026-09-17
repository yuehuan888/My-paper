"""把宽高比 > 1.9 的图从单栏 figure(\columnwidth) 提升为跨栏 figure*(\textwidth)。

为什么：这些图都是双面板宽图（宽高比 2.0-2.5）。放在单栏 3.5in 宽时
高度只有约 1.4in，双面板带坐标轴的图会小到看不清。跨栏后高度约 3.2in。

只动 figure 环境与 width，不改 caption/label。逐块处理并打印结果以便核对。
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "latex", "main.tex")

# 需要转跨栏的 label（按宽高比 > 1.9 判定）
WIDEN = [
    "fig:mech", "fig:power",
    "fig:e3", "fig:bound", "fig:bounddrift", "fig:e1",
    "fig:rep-arms", "fig:seed-conv", "fig:power-n10",
]

BLOCK = re.compile(
    r"\\begin\{figure\}(\[[^\]]*\])?\s*\n(.*?)\\label\{([^}]+)\}\s*\n\\end\{figure\}",
    re.S,
)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    src = io.open(TEX, encoding="utf-8").read()
    hits = []

    def repl(m):
        opt, body, label = m.group(1) or "[t]", m.group(2), m.group(3)
        if label not in WIDEN:
            return m.group(0)
        newbody = body.replace(r"width=\columnwidth", r"width=\textwidth")
        if newbody == body:
            print("  [警告] %s 未找到 width=columnwidth，仅改环境" % label)
        hits.append(label)
        return "\\begin{figure*}%s\n%s\\label{%s}\n\\end{figure*}" % (opt, newbody, label)

    out = BLOCK.sub(repl, src)

    print("已转跨栏 (%d): %s" % (len(hits), sorted(hits)))
    missing = [x for x in WIDEN if x not in hits]
    if missing:
        print("  [警告] 未匹配到: %s" % missing)

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(out)
    print("写入 %s" % TEX)
    print("figure  : %d" % len(re.findall(r"\\begin\{figure\}", out)))
    print("figure* : %d" % len(re.findall(r"\\begin\{figure\*\}", out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
