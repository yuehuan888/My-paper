"""修掉表格超宽（Overfull \\hbox）。

背景：tectonic 实测报出 8 个超宽表，单栏宽 252pt，超宽 7-96pt 不等。
其中 4 个是 v1.0 就存在的，4 个是 v2.0 新增的。

策略按超宽量分两档：
  · > 20pt：单栏塞不下，升为跨栏 table*（约 505pt 可用）
  · <= 20pt：留在栏内，只收窄列间距 \\tabcolsep 6pt -> 4pt
             （n 列表格省 2*(n-1)*2pt，5 列即 16pt）

只改环境与列间距，不动 caption / label / 数据。
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "latex", "main.tex")

# v2.1 新增的 §5.5 四张表（编译实测超宽 18-50pt）
WIDEN = ["tab:conv", "tab:levers", "tab:ceiling", "tab:redcnn90"]
# 已处理过的，不再动
TIGHTEN = []

# 注意：表格的 \label 紧跟在 \caption 之后，**不在** \end{table} 之前
# （图是反过来的）。所以先整块匹配，再从块内取 label。
BLOCK = re.compile(r"\\begin\{table\}(\[[^\]]*\])?\s*\n(.*?)\\end\{table\}", re.S)
LABEL = re.compile(r"\\label\{([^}]+)\}")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    src = io.open(TEX, encoding="utf-8").read()
    widened, tightened = [], []

    def repl(m):
        opt, body = m.group(1) or "[t]", m.group(2)
        lm = LABEL.search(body)
        label = lm.group(1) if lm else None
        if label in WIDEN:
            widened.append(label)
            return "\\begin{table*}%s\n%s\\end{table*}" % (opt, body)
        if label in TIGHTEN:
            tightened.append(label)
            if "\\setlength{\\tabcolsep}" in body:
                return m.group(0)
            new = body.replace("\\small\n", "\\small\n\\setlength{\\tabcolsep}{4pt}\n", 1)
            if new == body:
                new = body.replace("\\centering\n",
                                   "\\centering\n\\setlength{\\tabcolsep}{4pt}\n", 1)
            return "\\begin{table}%s\n%s\\end{table}" % (opt, new)
        return m.group(0)

    out = BLOCK.sub(repl, src)

    print("转跨栏 table* (%d): %s" % (len(widened), sorted(widened)))
    print("栏内收窄 (%d): %s" % (len(tightened), sorted(tightened)))
    for name, got in (("WIDEN", WIDEN), ("TIGHTEN", TIGHTEN)):
        miss = [x for x in got if x not in (widened + tightened)]
        if miss:
            print("  [警告] %s 未匹配: %s" % (name, miss))

    io.open(TEX, "w", encoding="utf-8", newline="\n").write(out)
    print("table  : %d" % len(re.findall(r"\\begin\{table\}", out)))
    print("table* : %d" % len(re.findall(r"\\begin\{table\*\}", out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
