"""投稿前的静态核查（无 LaTeX 工具链时的代偿，但比 check_latex.py 更严）。

检查项：
  1. 每个 \\includegraphics 的文件在 figures/ 下真实存在（路径拼写）
  2. 17 张图是否全部被嵌入（对照 figures/*.png）
  3. 每个 figure 环境是否有 \\label、每个 table 是否有 \\label
  4. figure* 是否与 \\begin{document} 匹配（跨栏图必须在双栏文档里）
  5. 摘要词数
  6. float 数量 vs \\extrafloats 配额
"""

from __future__ import annotations

import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEX = os.path.join(ROOT, "paper", "latex", "main.tex")
FIGDIR = os.path.join(ROOT, "paper", "figures")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    src = io.open(TEX, encoding="utf-8").read()
    body = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))
    ok = True

    print("=" * 74)
    print("[1] \\includegraphics 路径")
    print("=" * 74)
    inc = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body)
    missing = [p for p in inc if not os.path.exists(os.path.join(FIGDIR, p))]
    for p in inc:
        mark = "OK  " if os.path.exists(os.path.join(FIGDIR, p)) else "MISS"
        print("  [%s] %s" % (mark, p))
    if missing:
        print("  -> 缺失: %s" % missing)
        ok = False

    print()
    print("=" * 74)
    print("[2] 覆盖率：figures/*.png vs 已嵌入")
    print("=" * 74)
    on_disk = {os.path.basename(p) for p in glob.glob(os.path.join(FIGDIR, "*.png"))}
    embedded = set(inc)
    not_used = sorted(on_disk - embedded)
    print("  磁盘 %d 张，嵌入 %d 张" % (len(on_disk), len(embedded)))
    if not_used:
        print("  [提示] 有图未嵌入: %s" % not_used)
    extra = sorted(embedded - on_disk)
    if extra:
        print("  [FAIL] 嵌入了不存在的图: %s" % extra)
        ok = False

    print()
    print("=" * 74)
    print("[3] 浮动体 label 完整性")
    print("=" * 74)
    for env in ("figure", "figure*", "table", "table*"):
        blocks = re.findall(r"\\begin\{%s\}(.*?)\\end\{%s\}" % (re.escape(env), re.escape(env)),
                            body, re.S)
        nolabel = [b for b in blocks if "\\label{" not in b]
        if nolabel:
            print("  [FAIL] %s 中有 %d 个缺 \\label" % (env, len(nolabel)))
            ok = False
        else:
            print("  [OK  ] %s: %d 个，全部有 label" % (env, len(blocks)))

    print()
    print("=" * 74)
    print("[4] 跨栏图与文档类别")
    print("=" * 74)
    n_star = len(re.findall(r"\\begin\{figure\*\}", body))
    if n_star and "twocolumn" not in src and "conference" not in src:
        print("  [FAIL] 有 %d 个 figure* 但文档不是双栏" % n_star)
        ok = False
    else:
        print("  [OK  ] %d 个 figure*，documentclass 为 %s"
              % (n_star, re.search(r"\\documentclass(\[[^\]]*\])?\{([^}]+)\}", src).group(0)))

    print()
    print("=" * 74)
    print("[5] 摘要词数")
    print("=" * 74)
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", body, re.S).group(1)
    n_ab = len([w for w in ab.split() if re.search(r"[A-Za-z0-9]", w)])
    print("  %d 词（IEEE 会议常见上限 250，本文按完整记录保留两段）" % n_ab)

    print()
    print("=" * 74)
    print("[6] float 配额")
    print("=" * 74)
    nf = len(re.findall(r"\\begin\{(figure\*?|table\*?)\}", body))
    m = re.search(r"\\extrafloats\{(\d+)\}", src)
    quota = int(m.group(1)) if m else 0
    print("  float 共 %d 个，\\extrafloats = %d（默认上限约 18）" % (nf, quota))
    if nf > 18 + quota:
        print("  [FAIL] 超出配额")
        ok = False
    else:
        print("  [OK  ] 在配额内")

    print()
    print("=" * 74)
    print("结论:", "PASS" if ok else "FAIL")
    print("=" * 74)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
