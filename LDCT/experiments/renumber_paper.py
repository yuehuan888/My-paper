"""
把 conference_paper.md 的章节号整体后移一位，为新增的 §2 Related Work 腾位置。

为什么用脚本而不是手改
------------------------------------------------------------------
要改 17 个标题（## 2..6、### 2.1..4.3）加正文里的交叉引用。手改一次漏一处，
读者看到"见 §4"却翻到 §5，比不改还糟。脚本一次性做完并打印全部改动供核对。

顺序很关键：必须**从大到小**替换（6→7 先做），否则 2→3 之后 3→4 会把
刚改好的 3 又改成 4。

用法
    python experiments/renumber_paper.py --dry-run     # 只看会改什么
    python experiments/renumber_paper.py               # 真的改
"""

from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper", "conference_paper.md")

# 只匹配标题行。注意两种写法都要覆盖：
#   "## 2. Method"     —— 编号后带点
#   "### 2.1 Overview" —— 子编号后**不带**点
# 故末尾的点必须可选，否则子标题全部漏改（第一版就漏了 12 个）。
HEAD = re.compile(r"^(#{2,3}) (\d+)(?:\.(\d+))?\.?\s")


def shift_line(line: str):
    m = HEAD.match(line)
    if not m:
        return line, None
    hashes, major, minor = m.group(1), int(m.group(2)), m.group(3)
    if major < 2:                     # §1 Introduction 不动
        return line, None
    new = major + 1
    if minor is None:
        out = line.replace(f"{hashes} {major}.", f"{hashes} {new}.", 1)
    else:
        out = line.replace(f"{hashes} {major}.{minor}",
                           f"{hashes} {new}.{minor}", 1)
    return out, f"{m.group(0).strip()}  ->  {out.strip()}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    with open(PAPER, encoding="utf-8") as f:
        text = f.read()

    if "## 2. Related Work" in text:
        print("ℹ️  已经插入过 Related Work，无需重复。")
        return 0

    changes = []
    out_lines = []
    for line in text.split("\n"):
        new, ch = shift_line(line)
        out_lines.append(new)
        if ch:
            changes.append(ch)
    text2 = "\n".join(out_lines)

    # 正文交叉引用
    for old, new in [("(Section 4)", "(Section 5.2)"),
                     ("(Section 3.4)", "(Section 4.4)")]:
        if old in text2:
            n = text2.count(old)
            text2 = text2.replace(old, new)
            changes.append(f"正文交叉引用 {old} -> {new}  （{n} 处）")

    print(f"将改动 {len(changes)} 处：")
    for c in changes:
        print("   ", c)

    if args.dry_run:
        print("\n[dry-run] 未写入。")
        return 0

    with open(PAPER, "w", encoding="utf-8") as f:
        f.write(text2)
    print(f"\n✅ 已写入 {PAPER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
