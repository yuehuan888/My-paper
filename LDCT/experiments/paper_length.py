"""
论文篇幅估算 —— 投稿前的硬性检查。

为什么单独做：EI 会议对页数是**硬性**限制，且执行严格。
  · ICIP：正文 5 页 + 第 6 页**仅**参考文献（超页或第 6 页含非参考文献内容
    会直接拒审，连评审都不进）
  · EUSIPCO：含图、参考文献、附录**共 5 页**，超页"neither accepted, nor reviewed"
  · ICME：6 页含全部正文、图、参考文献

本脚本给的是**粗略估计**（IEEE 双栏正文约 1000 词/页），用于判断
"离上限还有多少余量"，不能替代用 LaTeX 模板排出来的真实页数。

用法
    python experiments/paper_length.py
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER = os.path.join(ROOT, "paper", "conference_paper.md")

LIMITS = {
    "ICIP 2027": "正文 5 页 + 第 6 页仅参考文献",
    "EUSIPCO 2027": "含图/参考文献/附录共 5 页",
    "ICME 2027": "6 页含全部正文/图/参考文献",
}


def wc(s: str) -> int:
    """去掉 markdown 记号后的词数（含数字）。"""
    s = re.sub(r"```.*?```", " ", s, flags=re.S)          # 代码块
    s = re.sub(r"[|#*_>`\[\]()]", " ", s)
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    with open(PAPER, encoding="utf-8") as f:
        txt = f.read()

    parts = txt.split("## References")
    body = parts[0]
    refs = parts[1] if len(parts) > 1 else ""
    n_refs = len(re.findall(r"^\[\d+\]", refs, re.M))

    # 章节词数分布
    secs = re.split(r"^#{2,3} ", body, flags=re.M)
    print(f"{'章节':<44}{'词数':>7}")
    print("-" * 53)
    for s in secs:
        if not s.strip():
            continue
        title = s.split("\n", 1)[0].strip()
        print(f"{title[:43]:<44}{wc(s):>7}")

    wb, wr = wc(body), wc(refs)
    print("-" * 53)
    print(f"{'正文合计':<44}{wb:>7}")
    print(f"{'参考文献':<44}{wr:>7}   ({n_refs} 条)")

    # IEEE 双栏经验值：
    #   正文 10pt 双栏 ≈ 1000 词/页
    #   参考文献 8pt，每条平均约 2 行，一栏约 55 行 → 约 27 条/栏 → 约 54 条/页
    REFS_PER_PAGE = 54
    print("\n【页数估算】IEEE 双栏：正文约 1000 词/页，参考文献约 54 条/页")
    print(f"  正文      ≈ {wb/1000:.1f} 页")
    print(f"  参考文献  ≈ {n_refs/REFS_PER_PAGE:.1f} 页   ({n_refs} 条)")
    print(f"  合计      ≈ {wb/1000 + n_refs/REFS_PER_PAGE:.1f} 页")
    print("\n  注：表格与图片会额外占版面。本文 4 张图 + 6 张表，")
    print("      实际排版后通常比纯词数估算多 0.5–1.0 页。")

    print("\n【各会限制对照】")
    for k, v in LIMITS.items():
        print(f"  {k:<14}{v}")

    print("\n⚠️  这是估算，不是排版结果。定稿前必须用目标会议的 LaTeX 模板")
    print("    实际排一遍，以模板给出的页数为准。")


if __name__ == "__main__":
    main()
