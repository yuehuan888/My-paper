"""收敛批分析：90 轮 vs 30 轮，以及同预算下的 RED-CNN 对照。

⚠️ 口径警告（这里踩过一次，写下来防复发）
================================================================
仓库里有**两批数据、同名文件、内容不同**：

    435 片（原始）    w3_*       三臂 n=5        redcnn_s0
    421 片（重新获取）fix_*     三臂 n=10       conv90_*（90 轮）
                      new_*                      redcnn90_*（90 轮）

拿 90 轮(421) 去比 w3_*(435) 会把「换数据」的效应算进「延长训练」里
（实测污染量约 ±0.03~0.06 dB，足以让 +0.33 这个结论偏掉）。
**30 轮的基线必须取 fix_*（同为 421 片）**，不是 w3_*。

本脚本因此不硬编码任何基线值，全部从 runs/ 现算。

用法
    python experiments/analyze_convergence.py
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def load(tag):
    rp = os.path.join(RUNS, tag, "results.json")
    if not os.path.exists(rp):
        return None
    r = json.load(io.open(rp, encoding="utf-8"))
    c = json.load(io.open(os.path.join(RUNS, tag, "config.json"), encoding="utf-8"))
    return {"tag": tag, "psnr": r["test_mean"]["PSNR"],
            "ssim": r["test_mean"].get("SSIM"), "n_test": c.get("n_test"),
            "epochs": c.get("epochs")}


def arm(prefix, name, seeds):
    out = []
    for s in seeds:
        r = load("%s_%s_s%d" % (prefix, name, s))
        if r:
            out.append(r)
    return out


def stats_of(rs):
    p = np.array([r["psnr"] for r in rs])
    s = np.array([r["ssim"] for r in rs if r["ssim"] is not None])
    return p, s


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ARMS = ["pr", "unconstrained", "fixed"]
    b30 = {a: arm("fix", a, range(10)) for a in ARMS}       # 421 片, 30 轮
    b90 = {a: arm("conv90", a, range(5)) for a in ARMS}     # 421 片, 90 轮
    b90_2 = arm("conv90", "bnd20", range(5))                # 421 片, 90 轮, β=2.0
    rc90 = []
    for s in range(5):
        r = load("redcnn90_s%d" % s)
        if r:
            rc90.append(r)

    # 口径自检：所有进入对比的 run 必须是同一个 n_test
    cohorts = {r["n_test"] for a in ARMS for r in b30[a] + b90[a]} | \
              {r["n_test"] for r in b90_2 + rc90}
    print("=" * 78)
    print("口径自检：所有参与对比的 run 的 n_test =", sorted(cohorts))
    if len(cohorts) != 1:
        print("  [FAIL] 口径不一致，下面的比较无效！")
        return 1
    print("  [OK  ] 口径一致（%d 片）" % cohorts.pop())
    print("=" * 78)

    print("\n[1] 三臂：30 轮 vs 90 轮（同为 421 片）")
    print("  %-16s %-24s %-24s %s" % ("臂", "30 轮", "90 轮", "变化"))
    for a in ARMS:
        p30, _ = stats_of(b30[a])
        p90, _ = stats_of(b90[a])
        print("  %-16s %7.4f ± %.4f (n=%d)   %7.4f ± %.4f (n=%d)   %+.4f"
              % (a, p30.mean(), p30.std(ddof=1), len(p30),
                 p90.mean(), p90.std(ddof=1), len(p90), p90.mean() - p30.mean()))

    print("\n[2] 差距的方向（这是局限 3 的直接答案）")
    print("  %-24s %-12s %-12s %s" % ("比较", "30 轮", "90 轮", "方向"))
    for a, b in (("pr", "fixed"), ("pr", "unconstrained"), ("unconstrained", "fixed")):
        g30 = stats_of(b30[a])[0].mean() - stats_of(b30[b])[0].mean()
        g90 = stats_of(b90[a])[0].mean() - stats_of(b90[b])[0].mean()
        print("  %-24s %+8.4f     %+8.4f     %s"
              % ("%s - %s" % (a, b), g30, g90,
                 "拉大" if abs(g90) > abs(g30) else "压缩"))

    print("\n[3] 90 轮上的配对检验")
    for a, b in (("pr", "unconstrained"), ("pr", "fixed"), ("unconstrained", "fixed")):
        xa, xb = stats_of(b90[a])[0], stats_of(b90[b])[0]
        if len(xa) != len(xb) or len(xa) < 2:
            continue
        d = xa - xb
        t, p = stats.ttest_rel(xa, xb)
        print("  %-26s Δ=%+.4f  p=%.4f  d_z=%+.2f"
              % ("%s - %s" % (a, b), d.mean(), p, d.mean() / d.std(ddof=1)))

    print("\n[4] 同预算（90 轮 / 421 片）下的 RED-CNN 对照")
    if not rc90:
        print("  (redcnn90_* 未完成)")
    else:
        prc, src = stats_of(rc90)
        print("  RED-CNN      n=%d  PSNR %.4f ± %.4f   SSIM %.4f"
              % (len(prc), prc.mean(), prc.std(ddof=1), src.mean() if len(src) else float("nan")))
        for lab, grp in (("PR-LWT β=0.5", b90["pr"]), ("PR-LWT β=2.0", b90_2)):
            if not grp:
                continue
            pg, _ = stats_of(grp)
            print("  %-12s n=%d  PSNR %.4f ± %.4f   距 RED-CNN %.4f dB"
                  % (lab, len(pg), pg.mean(), pg.std(ddof=1), prc.mean() - pg.mean()))
        print("\n  对照论文现值（30 轮 / 435 片，双方同预算）：1.0000 dB")

    print("\n[5] 训练稳定性（验证曲线极差）")
    for lab, pref, names in (("PR-LWT", "fix", ["pr", "fixed", "unconstrained"]),
                             ("RED-CNN", "redcnn90", [""])):
        rows = []
        for nm in names:
            for s in range(10):
                tag = ("%s_%s_s%d" % (pref, nm, s)) if nm else ("%s_s%d" % (pref, s))
                hp = os.path.join(RUNS, tag, "history.json")
                if not os.path.exists(hp):
                    continue
                v = json.load(io.open(hp, encoding="utf-8")).get("val_psnr", [])
                if len(v) >= 5:
                    rows.append(max(v) - min(v))
                if nm and s >= 4:
                    break
        if rows:
            print("  %-10s 验证极差 中位 %.3f dB  最大 %.3f dB  (n=%d 条曲线)"
                  % (lab, float(np.median(rows)), max(rows), len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
