"""抓取 Yuan1z0825/nature-skills 的精选子集（Apache-2.0）。

为什么不用 git clone / tarball / bash
================================================================
1. 本机 GitHub 经 Steam++ 加速器（hosts 劫持到 127.0.0.1），大传输会断：
   git clone 报 "early EOF"；codeload 的 tarball **没有 Content-Length**
   （服务端现打包），故 `curl -C -` 无法续传，会静默产出截断文件。
2. `raw.githubusercontent.com` 会**间歇性**返回 502（实测连续 2 次 502 后第 3 次 200），
   所以必须重试 + 退避。
3. **bash 版有路径翻译坑**：`mktemp -d` 给 bash 的是 `/tmp/tmp.XXX`，而 Windows 版
   Python 把 `/tmp/tmp.XXX` 解析成 `D:\tmp\tmp.XXX` —— 两个解释器对同一字符串的理解
   不同，中间文件会互相找不到。本脚本改用纯 Python 统一处理路径，杜绝此类问题。

为什么只装 5 个子技能
================================================================
该仓库面向 Nature 系期刊。其正文结构（宽读者叙事、Results/Discussion 分家）与
4–6 页的 ICIP/EUSIPCO/ICME 论文不兼容。只装**与期刊无关**的通用能力：
figure / statistics / ref-verifier / polishing / reviewer。

用法
================================================================
    python experiments/_fetch_nature_skills.py [目标目录]
默认目标：项目内 .claude/skills/（该路径已被 .gitignore 排除）
"""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import time

REPO = "Yuan1z0825/nature-skills"
BRANCH = "main"
SKILLS = ("nature-figure", "nature-statistics", "nature-ref-verifier",
          "nature-polishing", "nature-reviewer")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEST = os.path.join(HERE, "..", "..", ".claude", "skills")

# 本机 curl 必须带 --ssl-revoke-best-effort：加速器用的是自签根证书，
# schannel 的**吊销检查**会失败（CRYPT_E_NO_REVOCATION_CHECK）。
# 注意这只放宽 CRL 查询，**证书验证仍然开启**；不要用 -k / --insecure。
CURL = ["curl", "-sL", "--ssl-revoke-best-effort", "--max-time", "90"]


def curl_bytes(url: str, retries: int = 8, is_json: bool = False):
    """取 URL 内容，带指数退避重试。返回 bytes 或 None。"""
    for i in range(retries):
        try:
            p = subprocess.run(CURL + [url], capture_output=True, timeout=120)
            if p.returncode == 0 and p.stdout:
                # 502 的响应体很短（~52 字节的错误文本），不能当内容用
                if len(p.stdout) > 60:
                    return p.stdout
        except Exception:
            pass
        if i < retries - 1:
            time.sleep(min(4 * (i + 1), 30))
    return None


def fetch_tree() -> list:
    url = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}?recursive=1"
    raw = curl_bytes(url)
    if raw is None:
        sys.exit("FATAL: 无法取得仓库文件树（api.github.com 不通）")
    d = json.loads(raw.decode("utf-8"))
    if "tree" not in d:
        sys.exit(f"FATAL: tree API 返回异常: {str(d)[:300]}")
    return d["tree"]


def fetch_via_api(path: str):
    """兜底：走 api.github.com 的 contents 端点（base64）。故障域与 raw 不同。"""
    url = f"https://api.github.com/repos/{REPO}/contents/{path}?ref={BRANCH}"
    raw = curl_bytes(url, retries=4)
    if raw is None:
        return None
    try:
        d = json.loads(raw.decode("utf-8"))
        return base64.b64decode(d["content"])
    except Exception:
        return None


def main():
    dest = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DEST)
    print(f"目标目录: {dest}")

    tree = fetch_tree()
    keep = [e["path"] for e in tree
            if e.get("type") == "blob"
            and e["path"].startswith("skills/")
            and len(e["path"].split("/")) >= 3
            and e["path"].split("/")[1] in SKILLS]
    print(f"仓库共 {len(tree)} 个条目，匹配到 {len(keep)} 个文件")
    if not keep:
        sys.exit("FATAL: 未匹配到任何文件")

    n_ok = n_skip = n_fail = 0
    n_api = 0
    for i, path in enumerate(keep, 1):
        rel = path[len("skills/"):]                      # 去掉 skills/ 前缀
        out = os.path.join(dest, *rel.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            n_skip += 1
            continue

        raw = curl_bytes(
            f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}")
        if raw is None:
            raw = fetch_via_api(path)
            if raw is not None:
                n_api += 1

        if raw is None:
            n_fail += 1
            print(f"  [FAIL] {path}")
            continue

        with open(out, "wb") as f:
            f.write(raw)
        n_ok += 1
        if i % 25 == 0:
            print(f"  ... {i}/{len(keep)}  (ok={n_ok} skip={n_skip} fail={n_fail})")

    print(f"\n抓取完成: 新下 {n_ok}，已存在 {n_skip}，失败 {n_fail}"
          f"（其中走 API 兜底 {n_api}）")
    print("各技能落盘情况：")
    for s in SKILLS:
        d = os.path.join(dest, s)
        cnt = sum(len(f) for _, _, f in os.walk(d)) if os.path.isdir(d) else 0
        ok = os.path.exists(os.path.join(d, "SKILL.md"))
        print(f"  {'OK  ' if ok else 'MISS'} {s:<24} {cnt} 个文件")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
