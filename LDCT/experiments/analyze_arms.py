"""
三臂对照的统计分析与显著性检验。

⚠️ 独立单元是什么（这是本分析最关键的一点）
------------------------------------------------------------------
**独立单元是"训练种子"，不是"测试切片"。**

测试集的 435 张切片来自 2 个患者（或 L506 一个患者），同一患者的相邻脑切片
高度相关（层间距仅 2mm）。把它们当作 435 个独立样本做检验，会得到虚高若干
数量级的 p 值——这正是 MGF-Net 项目里踩过的"伪重复"陷阱。

故本脚本：
  - 以**种子**为独立单元（配对：同一种子跨臂）
  - 报告配对 t 检验与效应量
  - 同时给出 95% 置信区间
  - 明确标注 n 与检验的适用性边界

用法
------------------------------------------------------------------
    python experiments/analyze_arms.py                    # 默认划分
    python experiments/analyze_arms.py --prefix lit_      # 文献划分
    python experiments/analyze_arms.py --prefix w3_
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ARMS = ["pr", "unconstrained", "fixed"]


def collect(prefix: str):
    """返回 {arm: {seed: metrics}}"""
    out = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "results.json"))):
        d = os.path.basename(os.path.dirname(p))
        if not d.startswith(prefix):
            continue
        parts = d.split("_")
        # <prefix-without-underscore>_<arm>_s<seed>
        if len(parts) < 3:
            continue
        arm, seed = parts[-2], parts[-1].lstrip("s")
        if arm not in ARMS:
            continue
        try:
            seed = int(seed)
        except ValueError:
            continue
        r = json.load(open(p, encoding="utf-8"))
        out[arm][seed] = {
            "PSNR": r["test_mean"]["PSNR"],
            "SSIM": r["test_mean"]["SSIM"],
            "n_params": r["n_params"],
            "per_patient": r.get("test_per_patient", {}),
        }
    return out


def paired_ttest(a, b):
    """配对 t 检验。返回 (差值均值, t, 自由度, 近似 p, 95%CI)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b) or len(a) < 2:
        return None
    d = b - a
    n = len(d)
    mean = d.mean()
    sd = d.std(ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se if se > 1e-12 else float("inf")
    # 双尾 p（t 分布），用 scipy 若可用
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df=n - 1))
        crit = float(stats.t.ppf(0.975, df=n - 1))
    except Exception:  # noqa: BLE001
        p, crit = float("nan"), 1.96
    return {"mean": float(mean), "t": float(t), "df": n - 1, "p": p,
            "ci95": (float(mean - crit * se), float(mean + crit * se)),
            "sd": float(sd)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="w3_")
    ap.add_argument("--metric", default="PSNR", choices=["PSNR", "SSIM"])
    args = ap.parse_args()

    data = collect(args.prefix)
    have = [a for a in ARMS if data.get(a)]
    if not have:
        print(f"没有找到前缀 {args.prefix} 的结果")
        return

    seeds = sorted(set().union(*[set(data[a]) for a in have]))
    print("=" * 90)
    print(f"三臂对照分析  前缀={args.prefix}  指标={args.metric}")
    print("=" * 90)
    print(f"  参与比较的种子: {seeds}   (每臂每种子独立训练一次)")
    print()

    # 逐 run 明细
    print(f"  {'臂':<15}{'参数':>7}  " + "".join(f"{'s%d' % s:>9}" for s in seeds)
          + f"{'均值':>10}{'标准差':>9}")
    print("  " + "-" * 86)
    for a in have:
        vals, npar = [], None
        for s in seeds:
            v = data[a].get(s)
            vals.append(v[args.metric] if v else float("nan"))
            if v:
                npar = v["n_params"]
        arr = np.array([v for v in vals if not np.isnan(v)])
        print(f"  {a:<15}{npar:>7}  " + "".join(f"{v:>9.4f}" for v in vals)
              + f"{arr.mean():>10.4f}{arr.std(ddof=1):>9.4f}")
    print()

    # 配对检验
    print("  " + "-" * 86)
    print("  配对检验（独立单元 = 种子）")
    print("  " + "-" * 86)
    print(f"  {'对比':<34}{'差值':>10}{'t':>8}{'df':>5}{'p':>11}{'95% CI':>22}")
    for x, y in [("unconstrained", "pr"), ("fixed", "pr"),
                 ("fixed", "unconstrained")]:
        if x not in have or y not in have:
            continue
        common = [s for s in seeds if s in data[x] and s in data[y]]
        if len(common) < 2:
            continue
        r = paired_ttest([data[x][s][args.metric] for s in common],
                         [data[y][s][args.metric] for s in common])
        if r is None:
            continue
        ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
        print(f"  {y} - {x:<26}{r['mean']:>+10.4f}{r['t']:>8.2f}{r['df']:>5}"
              f"{r['p']:>11.4f}{ci:>22}")

    print()
    print("  ⚠️ 边界说明")
    print(f"     · 独立单元是**种子**（n={len(seeds)}），不是测试切片。")
    print("       435 张切片来自 2 个患者，相邻切片高度相关，")
    print("       把它们当独立样本会得到虚高的显著性——这是伪重复。")
    print(f"     · n={len(seeds)} 偏小，p 值仅作参考；效应量（差值/标准差）更稳健。")
    print("     · 若需更强证据，应增加种子数而非增加测试切片。")

    # 逐患者
    print()
    print("  " + "-" * 86)
    print("  逐患者明细（每臂取各种子均值）")
    print("  " + "-" * 86)
    pats = set()
    for a in have:
        for s in data[a]:
            pats.update(data[a][s]["per_patient"].keys())
    for pid in sorted(pats):
        line = f"  {pid:<8}"
        for a in have:
            v = [data[a][s]["per_patient"][pid]["PSNR"]
                 for s in seeds if s in data[a] and pid in data[a][s]["per_patient"]]
            line += f"{np.mean(v):>12.4f}" if v else f"{'—':>12}"
        print(line + f"   {args.metric}（各臂跨种子均值）")
    print()


if __name__ == "__main__":
    main()
