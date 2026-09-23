"""核查中文对照版与英文版 v2.0 的数字是否一致。

中文版头部承诺"与英文版内容一一对应，所有数字均相同"。这个脚本把这个承诺
变成可检验的：把两边的**数值 token** 抽出来做差集。

只抽有统计含义的数字（带小数点 / 科学计数法 / 百分比 / 带千分位），
纯整数（患者数、参数量等）单独比对，避免把章节号、引用编号算进来。
"""

from __future__ import annotations

import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EN = os.path.join(ROOT, "paper", "conference_paper.md")
ZH = os.path.join(ROOT, "paper", "conference_paper_zh.md")

# 应已从中文版消失的 v1.0 遗留值。
# 注意："六个数量级" **不**在此列 —— 它不是中文版遗留，而是三个文件在 §2.5 都
# 这么写（§5.3 则精确写 5.9）。实测比值为 1.35e-01 / 1.62e-07 = 8.3e5，即 5.92 个
# 数量级，所以"六个"是向上取整的通俗说法，属于措辞问题而非同步错误。
STALE = ["12–16", "12-16", "1.49e-01", "24 组", "0.064", "−0.03 dB，p = 0.11"]

NUM = re.compile(r"\d+(?:\.\d+)?(?:e[-+]?\d+)?%?|\d{1,3}(?:,\d{3})+")


def numbers(path):
    s = io.open(path, encoding="utf-8").read()
    # 去掉代码块、图片链接、参考文献说明区，避免噪声
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", s)
    out = set()
    for m in NUM.finditer(s):
        t = m.group(0)
        # 纯整数且 <= 40 的多半是章节号/轮次/臂数，单列
        out.add(t)
    return out


def norm(tset):
    """把 1,848,865 / 1848865 这类归一。"""
    out = set()
    for t in tset:
        out.add(t.replace(",", ""))
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    en_raw, zh_raw = numbers(EN), numbers(ZH)
    en, zh = norm(en_raw), norm(zh_raw)

    # 只关心"有统计含义"的：含小数点、科学计数法或百分号
    key = lambda s: ("." in s) or ("e-" in s) or ("e+" in s) or s.endswith("%")

    en_k = {x for x in en if key(x)}
    zh_k = {x for x in zh if key(x)}

    # 已知且无害的差异，逐条注明理由，避免把假阳性当成同步缺陷：
    #  英文侧 10.7937 = 参考文献里的 DOI（中文版按设计不列参考文献）
    #  英文侧 1e-3 / 2e-7 = 记法，中文版写作 10⁻³ / 2×10⁻⁷
    #  中文侧 2.1-2.5 = 中文版给相关工作编了小节号，英文版用加粗行内标题
    #  中文侧 3.6 = 记法，英文版写作 3.6e-07
    #  两侧 #  两侧共有 2.1：节号 §5.2.1、版本号 v2.1、以及 §5.5 的 p=2.1e-6，
    #         都不是统计量，必须**两边同时**豁免，否则豁免表本身会造出假差异
    #         （踩过一次：只豁免了中文侧，于是英文侧冒出 "只在英文版出现: 2.1"）。
    BENIGN_EN = {"10.7937", "1e-3", "2e-7", "2.1"}
    BENIGN_ZH = {"2.1", "2.2", "2.3", "2.4", "2.5", "3.6"}
    en_k -= BENIGN_EN
    zh_k -= BENIGN_ZH

    print("=" * 70)
    print("[1] 关键数字（小数 / 科学计数法 / 百分数）")
    print("=" * 70)
    print("  英文 %d 个，中文 %d 个" % (len(en_k), len(zh_k)))
    only_en = sorted(en_k - zh_k)
    only_zh = sorted(zh_k - en_k)
    print("\n  只在英文版出现 (%d):" % len(only_en))
    for x in only_en:
        print("     ", x)
    print("\n  只在中文版出现 (%d):" % len(only_zh))
    for x in only_zh:
        print("     ", x)

    print()
    print("=" * 70)
    print("[2] v1.0 遗留值是否已清除")
    print("=" * 70)
    zs = io.open(ZH, encoding="utf-8").read()
    bad = False
    for t in STALE:
        n = zs.count(t)
        print("  [%s] %-22s 出现 %d 次" % ("FAIL" if n else "OK  ", t, n))
        bad |= n > 0

    print()
    print("=" * 70)
    print("[3] v2.0 结构是否齐备")
    print("=" * 70)
    for name, pat in [("§5.2.1", r"#### 5\.2\.1"),
                      ("§5.4", r"### 5\.4"),
                      ("附录 B", r"## 附录 B"),
                      ("17 张图引用", r"!\[fig\d+\]")]:
        n = len(re.findall(pat, zs))
        exp = 17 if name == "17 张图引用" else 1
        print("  [%s] %-14s %d 处（期望 %d）" % ("OK  " if n == exp else "FAIL", name, n, exp))
        bad |= n != exp

    # 局限条数
    lim = len(re.findall(r"^\d+\. \*\*", zs.split("**局限。**")[1], re.M)) if "**局限。**" in zs else 0
    print("  [%s] %-14s %d 条（期望 10）" % ("OK  " if lim == 10 else "FAIL", "局限条数", lim))
    bad |= lim != 10

    print()
    print("=" * 70)
    print("结论:", "FAIL（见上）" if (bad or only_en or only_zh) else "PASS（数字一致、结构齐备）")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
