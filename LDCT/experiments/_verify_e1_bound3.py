"""临时核查 3：E1 容量表按 n_conv 分组重算；并查各 bound 批的数据口径(n_test)。"""

from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def cfg_of(tag):
    p = os.path.join(HERE, "runs", tag, "config.json")
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def res_of(tag):
    p = os.path.join(HERE, "runs", tag, "results.json")
    if not os.path.exists(p):
        return None
    return json.load(io.open(p, encoding="utf-8"))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 78)
    print("A) 各批数据口径（n_test / split），用于判断能否并表")
    print("=" * 78)
    seen = {}
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        tag = os.path.basename(d)
        c = cfg_of(tag)
        if not c:
            continue
        pref = tag.rsplit("_s", 1)[0]
        seen.setdefault(pref, set()).add((c.get("n_test"), c.get("n_train"),
                                          os.path.basename(str(c.get("split_file")))))
    for pref in sorted(seen):
        if any(k in pref for k in ("bnd", "pr", "fixed", "unconstrained", "e1", "e3")):
            print("  %-28s %s" % (pref, sorted(seen[pref])))

    print()
    print("=" * 78)
    print("B) E1 容量扫描（按 n_conv 分组，跨 wavelet 比较）")
    print("=" * 78)
    groups = {}
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        tag = os.path.basename(d)
        if not tag.startswith("e1_"):
            continue
        c, r = cfg_of(tag), res_of(tag)
        if not (c and r):
            continue
        kind = ("pr" if "_pr_" in tag else
                "fixed" if "_fixed_" in tag else
                "unconstrained" if "_unconstrained_" in tag else None)
        if kind is None:
            continue
        groups.setdefault(c["n_conv"], {}).setdefault(kind, []).append(r["test_mean"]["PSNR"])

    print("  %-8s %-8s %-22s %-22s %s" % ("n_conv", "params", "PR", "fixed", "Δ = PR − fixed"))
    for nc in sorted(groups):
        g = groups[nc]
        pr, fx = np.array(g.get("pr", [])), np.array(g.get("fixed", []))
        if len(pr) and len(fx):
            print("  %-8d %-8s n=%-3d %-14s n=%-3d %-14s %+.4f"
                  % (nc, "", len(pr), "%.4f" % pr.mean(), len(fx), "%.4f" % fx.mean(),
                     pr.mean() - fx.mean()))
        if "unconstrained" in g:
            un = np.array(g["unconstrained"])
            print("           unconstrained n=%d mean=%.4f" % (len(un), un.mean()))

    print()
    print("  注：E1 的 n_conv=2 档即主实验默认容量，取自主 three-arm 批。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
