"""LaTeX 结构自检 —— 本机没有 LaTeX 工具链时的代偿。

为什么需要
================================================================
本机**没有** tectonic / pdflatex / xelatex（实测 which 全空），
所以改完 `main.tex` 无法用编译器验证。但"结构是否自洽"是可以静态查的，
而这一类错误（环境不配对、花括号失衡、占位符残留）恰恰是手改最容易引入的。

它**不能**替代真正的编译（宏展开、宏包兼容、排版溢出都查不出来），
只用来在提交前挡掉低级错误。**定稿前仍必须在有工具链的机器上排一遍。**

用法
================================================================
    python experiments/check_latex.py paper/latex/main.tex
"""

from __future__ import annotations

import collections
import io
import re
import sys


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "paper/latex/main.tex"
    src = io.open(path, encoding="utf-8").read()

    # 去掉整行注释（注意：行内 % 转义为 \% 的不算注释，这里只处理行首）
    body = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith("%"))

    ok = True

    print("=" * 70)
    print(f"LaTeX 结构自检: {path}")
    print("=" * 70)

    # ---------------- 环境配对 ----------------
    begins = collections.Counter(re.findall(r"\\begin\{([^}]+)\}", body))
    ends = collections.Counter(re.findall(r"\\end\{([^}]+)\}", body))
    print("\n[1] 环境配对")
    for env in sorted(set(begins) | set(ends)):
        b, e = begins[env], ends[env]
        good = (b == e)
        ok &= good
        print(f"  [{'OK  ' if good else 'FAIL'}] {env:<18} begin={b} end={e}")

    # ---------------- 花括号 ----------------
    # 排除 \{ \} 这两个转义符
    lb = body.count("{") - body.count("\\{")
    rb = body.count("}") - body.count("\\}")
    print(f"\n[2] 花括号平衡: {{={lb}  }}={rb}  ", end="")
    if lb == rb:
        print("OK")
    else:
        print(f"FAIL (差 {lb - rb})")
        ok = False

    # ---------------- 行内数学 $ ----------------
    # ⚠️ 必须**整体**统计，不能逐行：LaTeX 的行内数学可以跨行，
    #    逐行查会把 "$...$" 跨行的正常写法误报成"奇数个 $"（实测踩过）。
    #    代价是丢失定位信息，故只在总数为奇数时才逐行提示可能的位置。
    n_dollar = len(re.findall(r"(?<!\\)\$", body))
    print(f"\n[3] 行内数学 $ 总数: {n_dollar}  ", end="")
    if n_dollar % 2 == 0:
        print("OK（偶数）")
    else:
        print("FAIL（奇数，必有未闭合的 $）")
        ok = False
        print("    逐行累计值（在跳跃处附近找）：")
        run = 0
        for i, line in enumerate(body.split("\n"), 1):
            run += len(re.findall(r"(?<!\\)\$", line))
            if run % 2 == 1 and i % 20 == 0:
                print(f"      第 {i} 行后累计 {run}（仍处于数学模式）: {line.strip()[:60]}")

    # ---------------- 占位符 / 非法环境 ----------------
    print("\n[4] 占位符与非法构造")
    hits = False
    for pat in ("comment_old", "TODO", "FIXME", "XXX", "PLACEHOLDER", "???", "\\\\todo"):
        n = body.count(pat)
        if n:
            print(f"  [FAIL] {pat} 出现 {n} 次")
            hits = True
    if not hits:
        print("  OK（无残留）")

    # ---------------- 引用完整性 ----------------
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
    # `\cite{a,b,%\n  c}` 这种跨行 + 行内注释的写法很常见，先把 `%` 到行尾删掉
    # 再提取 key，否则会把 "%\nc" 当成一个不存在的 key（实测误报过）。
    no_comment = re.sub(r"%(?!\\)[^\n]*", "", body)
    cites = set()
    for m in re.findall(r"\\cite\{([^}]*)\}", no_comment, flags=re.S):
        cites.update(c.strip() for c in m.split(",") if c.strip())
    bibitems = set(re.findall(r"\\bibitem\{([^}]+)\}", body))

    print("\n[5] 交叉引用")
    dangling = refs - labels
    if dangling:
        print(f"  [FAIL] \\ref 指向不存在的 label: {sorted(dangling)}")
        ok = False
    else:
        print(f"  OK（{len(refs)} 个 \\ref 全部有对应 \\label）")

    unused = labels - refs
    if unused:
        print(f"  [提示] 定义了但未被引用的 label ({len(unused)}): {sorted(unused)[:6]}")

    print("\n[6] 引用")
    if bibitems:
        missing = cites - bibitems
        if missing:
            print(f"  [FAIL] \\cite 了但没有 \\bibitem: {sorted(missing)}")
            ok = False
        else:
            print(f"  OK（{len(cites)} 个 cite 全部有对应 bibitem）")
        uncited = bibitems - cites
        if uncited:
            print(f"  [提示] 有 bibitem 但从未被 cite ({len(uncited)}): {sorted(uncited)[:8]}")
    else:
        print("  (未发现 \\bibitem，可能用 BibTeX/外部 .bib，跳过)")

    print("\n" + "=" * 70)
    print("结论:", "PASS（结构自洽）" if ok else "FAIL（见上）")
    print("⚠️ 这**不能替代编译**：宏展开、宏包兼容、排版溢出都查不出来。")
    print("=" * 70)
    return 1 if not ok else 0


if __name__ == "__main__":
    sys.exit(main())
