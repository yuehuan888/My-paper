"""临时探查：确认 runs/ 里逐切片数据的可用性与对齐假设。只读，不落盘。

要确认的三件事：
1. test_per_sample 的顺序是否与 test_per_patient 的分组一致（能否拆出患者标签）
2. 不同运行之间逐切片顺序是否可比（能否做配对逐切片检验）
3. history.json 是否每次运行都完整
"""
import io
import json
import glob
import os
import sys

import numpy as np

RUNS = os.path.join(os.path.dirname(__file__), "runs")


def load(tag):
    p = os.path.join(RUNS, tag, "results.json")
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    # ---------- 1. 顺序假设：前 n1 条是否就是第一个患者 ----------
    d = load("w3_pr_s0")
    per = d["test_per_patient"]
    order = list(per.keys())
    ps = np.array([r["PSNR"] for r in d["test_per_sample"]])
    print("患者顺序:", order)
    print("总切片:", len(ps))
    off = 0
    for name in order:
        n = per[name]["n"]
        seg = ps[off:off + n]
        print("  %-6s n=%3d  段均值=%.6f  报告值=%.6f  差=%.2e"
              % (name, n, seg.mean(), per[name]["PSNR"],
                 abs(seg.mean() - per[name]["PSNR"])))
        off += n
    print("  段长合计 =", off)

    # ---------- 2. 跨运行对齐：所有 S1 运行是否同长 ----------
    print("\n各运行逐切片条数:")
    for tag in sorted(os.listdir(RUNS)):
        f = os.path.join(RUNS, tag, "results.json")
        if not os.path.exists(f):
            continue
        dd = load(tag)
        tp = dd.get("test_per_sample", [])
        hist = os.path.join(RUNS, tag, "history.json")
        ne = len(json.load(io.open(hist, encoding="utf-8"))["epoch"]) if os.path.exists(hist) else -1
        print("  %-22s n=%4d  epochs=%3d  patients=%s"
              % (tag, len(tp), ne, list(dd.get("test_per_patient", {}).keys())))

    # ---------- 3. 配对性验证：同一臂不同种子的逐切片顺序应一致 ----------
    # 若顺序一致，则 (seed_a 的 slice i) 与 (seed_b 的 slice i) 是同一张切片。
    # 无法直接验证标签，但可用"邻近切片高度相关"这一先验做间接检验：
    # 打乱顺序会破坏相邻差分的分布。
    print("\n相邻切片差分（自相关代理）——顺序若被打乱应显著变大:")
    for tag in ["w3_pr_s0", "w3_pr_s1", "w3_unconstrained_s0", "w3_fixed_s0"]:
        v = np.array([r["PSNR"] for r in load(tag)["test_per_sample"]])
        adj = np.abs(np.diff(v)).mean()
        rng = np.random.RandomState(0)
        shuff = np.abs(np.diff(v[rng.permutation(len(v))])).mean()
        print("  %-22s 相邻=%.4f  打乱后=%.4f  比值=%.2f"
              % (tag, adj, shuff, shuff / max(adj, 1e-9)))


if __name__ == "__main__":
    main()
