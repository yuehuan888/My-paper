"""临时：估算 main.tex 正文词数（与 paper_length.py 的口径对齐，便于比较）。"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "paper", "latex", "main.tex")

BS = chr(92)


def strip_tex(s: str) -> str:
    # 去掉整行注释
    s = "\n".join(l for l in s.split("\n") if not l.lstrip().startswith("%"))
    # 去掉环境起止
    s = re.sub(BS + BS + r"(begin|end)\{[a-z*]+\}(\[[^\]]*\])?(\{[^}]*\})?", " ", s)
    # 去掉命令名，保留其参数文本（\textbf{x} -> x）
    s = re.sub(BS + BS + r"[a-zA-Z]+\*?(\[[^\]]*\])?", " ", s)
    s = s.replace("{", " ").replace("}", " ")
    return s


def wc(s: str) -> int:
    return len([w for w in s.split() if re.search(r"[A-Za-z0-9]", w)])


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    s = io.open(TEX, encoding="utf-8").read()
    body = s.split(BS + "begin{thebibliography}")[0]
    n = wc(strip_tex(body))
    print(f"main.tex 正文（含表格/图注文本）: {n} 词")
    print(f"  ≈ {n/1000:.1f} 页纯文字（IEEE 双栏 1000 词/页）")
    print(f"  环境计数: figure={body.count(BS+'begin{figure}')+body.count(BS+'begin{figure*}')}"
          f"  table={body.count(BS+'begin{table}')+body.count(BS+'begin{table*}')}")


if __name__ == "__main__":
    main()
