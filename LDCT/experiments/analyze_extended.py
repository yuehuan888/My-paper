"""扩展分析：从**已有产物**中榨出论文还没用到的证据（零算力）。

为什么存在
================================================================
论文 v1.0 只用了均值±标准差和三个配对 t 检验。但 `runs/` 里其实存着
**逐切片** PSNR/SSIM（435 或 211 条）与**每轮**训练曲线（30 轮），
这些数据没有被任何图表使用过。

本脚本不训练任何东西，只读 `runs/*/{results,history}.json`。

统计立场（必须遵守，不能放松）
================================================================
论文明确以**训练种子**为独立单元（测试集只有 2 个患者，切片高度相关）。
故本脚本：

  - 显著性检验一律在**种子层面**做（n=5 或 3）
  - 逐切片数据**只用于描述性统计**（分布、离散度、重叠度、逐患者分解）
  - **不**把 435 张切片当独立样本做检验 —— 那会把显著性虚高几个数量级

对齐假设（已实测确认，非推测）
================================================================
`_explore_artifacts.py` 实测：
  - `test_per_sample` 前 211 条 = L506，其后 224 条 = L067（段均值与
    `test_per_patient` 报告值**逐位相同**，差 0.00e+00）
  - 所有 S1 运行都是 435 条、所有 S2 运行都是 211 条
  - 相邻切片 PSNR 差分 0.267 vs 打乱后 2.18（**8.2 倍**）→ 顺序确实
    按切片保持，跨运行的逐切片配对是有意义的

输出
================================================================
  - `analysis_extended.json`  —— 全部数字
  - `../paper/figures/fig5_*.png` ~ `fig9_*.png`
"""

from __future__ import annotations

import io
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FIGDIR = os.path.join(HERE, "..", "paper", "figures")
OUT_JSON = os.path.join(HERE, "analysis_extended.json")

# 三臂配色（色盲友好；pr 为本文方法）
COLORS = {"pr": "#2E5EAA", "unconstrained": "#D1495B", "fixed": "#6C757D"}
LABELS = {
    "pr": "PR-LWT (ours)",
    "unconstrained": "Unconstrained",
    "fixed": "Fixed Haar",
}
# 每个划分下，各臂的运行 tag 前缀与种子数
ARMS = {
    "S1": {"prefix": "w3_", "seeds": range(5), "n_slices": 435,
           "patients": ["L506", "L067"]},
    "S2": {"prefix": "lit_", "seeds": range(3), "n_slices": 211,
           "patients": ["L506"]},
}
PAT_N = {"L506": 211, "L067": 224}


# ---------------------------------------------------------------- 读取
def load_json(path):
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def load_run(tag):
    """返回 (per_slice_psnr, per_slice_ssim, per_patient, history) 或 None。"""
    rd = os.path.join(RUNS, tag)
    rp = os.path.join(rd, "results.json")
    if not os.path.exists(rp):
        return None
    d = load_json(rp)
    hp = os.path.join(rd, "history.json")
    hist = load_json(hp) if os.path.exists(hp) else None
    ps = np.array([r["PSNR"] for r in d.get("test_per_sample", [])], dtype=float)
    ss = np.array([r["SSIM"] for r in d.get("test_per_sample", [])], dtype=float)
    return {
        "tag": tag, "psnr": ps, "ssim": ss,
        "per_patient": d.get("test_per_patient", {}),
        "mean": d.get("test_mean", {}),
        "n_params": d.get("n_params"),
        "best_epoch": d.get("best_epoch"),
        "history": hist,
    }


def seed_mean_series(split, arm):
    """某个 (划分, 臂) 下，每种子的测试集均值 PSNR —— 显著性检验的输入。"""
    cfg = ARMS[split]
    out = []
    for s in cfg["seeds"]:
        r = load_run("%s%s_s%d" % (cfg["prefix"], arm, s))
        if r is not None:
            out.append(r["mean"]["PSNR"])
    return np.array(out)


def seed_psr_matrix(split, arm):
    """(n_seeds, n_slices) 的逐切片 PSNR 矩阵。"""
    cfg = ARMS[split]
    rows, tags = [], []
    for s in cfg["seeds"]:
        r = load_run("%s%s_s%d" % (cfg["prefix"], arm, s))
        if r is not None:
            rows.append(r["psnr"])
            tags.append(r["tag"])
    return np.vstack(rows) if rows else np.empty((0, 0)), tags


# ------------------------------------------------------------ 统计工具
def convergence_stats(split, arm):
    """从 history.json 提取收敛特征。

    论文只报了测试集终值，从未分析**训练过程**。但"PR 让优化更容易/更稳定"
    本身可能是一个独立可写的卖点，而且答案全在已有的 history.json 里。
    """
    cfg = ARMS[split]
    curves, best_eps = [], []
    for s in cfg["seeds"]:
        r = load_run("%s%s_s%d" % (cfg["prefix"], arm, s))
        if r is None or not r["history"]:
            continue
        h = r["history"]
        curves.append({
            "train_l1": np.array(h["train_l1"], float),
            "val_psnr": np.array(h["val_psnr"], float),
            "epoch": np.array(h["epoch"], float),
        })
        best_eps.append(r["best_epoch"])
    if not curves:
        return None

    # ⚠️ train_l1 每轮都有（30 条），val_psnr 只有验证轮有（6 条，val_interval=5）。
    #    两者**长度不同**，绝不能按 min(len) 截断对齐——那会把训练曲线砍到 6 轮，
    #    并把 6 个验证点画在 x=1..6（时间轴压缩 5 倍）。实测踩过这个坑。
    n_ep = min(len(c["train_l1"]) for c in curves)
    n_val = min(len(c["val_psnr"]) for c in curves)
    L1 = np.vstack([c["train_l1"][:n_ep] for c in curves])
    VP = np.vstack([c["val_psnr"][:n_val] for c in curves])

    # 复原验证轮次：train.py 的规则是 `ep % val_interval == 0 or ep == epochs`。
    # 旧 config.json 没记 val_interval，故按实际点数反推。
    cand = [vi for vi in range(1, n_ep + 1)
            if len([e for e in range(1, n_ep + 1) if e % vi == 0 or e == n_ep]) == n_val]
    val_interval = max(cand) if cand else max(1, n_ep // max(n_val, 1))
    val_epochs = [e for e in range(1, n_ep + 1)
                  if e % val_interval == 0 or e == n_ep][:n_val]

    return {
        "n_seeds": len(curves),
        "n_epochs": int(n_ep),
        "n_val_points": int(n_val),
        "val_interval": int(val_interval),
        "val_epochs": [int(e) for e in val_epochs],
        "train_epochs": list(range(1, n_ep + 1)),
        "train_l1_curve_mean": [float(v) for v in L1.mean(0)],
        "train_l1_curve_sd": [float(v) for v in L1.std(0, ddof=1)] if len(curves) > 1 else [0.0] * n_ep,
        "val_psnr_curve_mean": [float(v) for v in VP.mean(0)],
        "val_psnr_curve_sd": [float(v) for v in VP.std(0, ddof=1)] if len(curves) > 1 else [0.0] * n_val,
        "final_train_l1": float(L1[:, -1].mean()),
        "best_val_psnr": float(VP.max(1).mean()),
        "best_epoch_each": [int(e) for e in best_eps],
        # 训练末段（后 20%）的逐轮波动 —— 衡量收敛后是否还在抖
        "late_epoch_instability": float(L1[:, int(n_ep * 0.8):].std(axis=1, ddof=1).mean()),
    }


def variance_test(split):
    """三臂**种子间**方差是否齐性（Levene，中位数中心，对非正态稳健）。

    `pr` 臂的种子间 sd（0.0815）明显大于另两臂（0.03 量级）。
    这本身是个诚实要报的现象，但 n=5 下能否断言"方差不同"必须做检验，
    不能凭数字大小下结论。
    """
    from scipy import stats
    groups = []
    for arm in ("pr", "unconstrained", "fixed"):
        v = seed_mean_series(split, arm)
        if len(v) > 1:
            groups.append(v)
    if len(groups) < 2:
        return None
    stat, p = stats.levene(*groups, center="median")
    return {"levene_stat": float(stat), "levene_p": float(p),
            "sds": [float(g.std(ddof=1)) for g in groups]}


def paired_t(a, b):
    """配对 t 检验（独立单元 = 种子）。返回 (delta, t, p, ci_lo, ci_hi)。"""
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    d = a - b
    delta = d.mean()
    if n < 2 or d.std(ddof=1) == 0:
        return delta, float("nan"), float("nan"), float("nan"), float("nan")
    t, p = stats.ttest_rel(a, b)
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    return float(delta), float(t), float(p), float(delta - crit * se), float(delta + crit * se)


def cohens_dz(a, b):
    """配对 Cohen's d_z —— 效应量，与样本量无关。"""
    d = np.asarray(a, float) - np.asarray(b, float)
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else float("nan")


def bootstrap_ci(x, stat=np.mean, n_boot=10000, seed=0):
    """对给定样本做 bootstrap 的 95% CI。"""
    rng = np.random.RandomState(seed)
    x = np.asarray(x, float)
    bs = [stat(x[rng.randint(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def mdes(n, sd, alpha=0.05, power=0.80):
    """配对设计的**最小可检测效应量**（占标准差倍数）。

    检验的是「效应量 / 标准差」这个比值能有多大 —— 与单位无关。
    用于回答"n=5 到底够不够"，以及"没检出差异 ≠ 没有差异"。
    """
    from scipy import stats
    if n < 2:
        return float("nan")
    t_crit = stats.t.ppf(1 - alpha / 2, n - 1)
    t_beta = stats.t.ppf(power, n - 1)
    return float((t_crit + t_beta) / np.sqrt(n))


# ------------------------------------------------------------ 分析
def analyse():
    R = {"splits": {}, "roundtrip": {}}

    for split in ("S1", "S2"):
        cfg = ARMS[split]
        S = {"arms": {}, "comparisons": {}, "power": {}}

        for arm in ("pr", "unconstrained", "fixed"):
            sm = seed_mean_series(split, arm)
            M, tags = seed_psr_matrix(split, arm)
            if len(sm) == 0:
                continue
            # 逐患者（各种子平均）
            per_pat = {}
            off = 0
            for p in cfg["patients"]:
                n = PAT_N[p]
                per_pat[p] = {
                    "n": n,
                    "over_seeds_mean": float(M[:, off:off + n].mean()),
                    "over_seeds_sd_of_seedmeans": float(M[:, off:off + n].mean(1).std(ddof=1))
                    if M.shape[0] > 1 else 0.0,
                }
                off += n
            S["arms"][arm] = {
                "tags": tags,
                "n_seeds": len(sm),
                "seed_means": [float(v) for v in sm],
                "mean": float(sm.mean()),
                "sd": float(sm.std(ddof=1)) if len(sm) > 1 else 0.0,
                "sem": float(sm.std(ddof=1) / np.sqrt(len(sm))) if len(sm) > 1 else 0.0,
                # 逐切片描述统计（**只作描述，不做检验**）
                "slice_mean": float(M.mean()),
                "slice_p05": float(np.percentile(M, 5)),
                "slice_p95": float(np.percentile(M, 95)),
                # 每种子内部的切片离散度之平均 —— 衡量"单张片子的波动"
                "within_seed_sd": float(M.std(axis=1, ddof=1).mean()),
                # 种子间在**同一张切片**上的离散度之平均 —— 衡量"训练随机性"
                "across_seed_sd": float(M.std(axis=0, ddof=1).mean()) if M.shape[0] > 1 else 0.0,
                "per_patient": per_pat,
                "bs_ci": bootstrap_ci(sm) if len(sm) > 1 else (float("nan"), float("nan")),
            }

        # 配对比较（种子层面）
        for a, b in (("pr", "unconstrained"), ("pr", "fixed"), ("unconstrained", "fixed")):
            xa, xb = seed_mean_series(split, a), seed_mean_series(split, b)
            if len(xa) == 0 or len(xa) != len(xb):
                continue
            d, t, p, lo, hi = paired_t(xa, xb)
            S["comparisons"]["%s-%s" % (a, b)] = {
                "delta": d, "t": t, "p": p, "ci": [lo, hi],
                "cohens_dz": cohens_dz(xa, xb),
                "n": int(len(xa)),
                # 差值相对种子间波动的倍数（论文用过"8×标准差"这个说法）
                "delta_over_sd": float(abs(d) / xa.std(ddof=1)) if xa.std(ddof=1) > 0 else float("nan"),
            }

        # 收敛动力学（论文未用过的维度）
        S["convergence"] = {}
        for arm in ("pr", "unconstrained", "fixed"):
            cs = convergence_stats(split, arm)
            if cs:
                S["convergence"][arm] = cs

        # 种子间方差齐性
        S["variance_test"] = variance_test(split)

        # 功效：n=5/3 下能检出的最小效应（以种子间标准差为单位）
        base_sd = S["arms"]["pr"]["sd"] if "pr" in S["arms"] else 0.05
        for n in (3, 5, 8, 10, 20):
            S["power"]["n=%d" % n] = {
                "mdes_in_sd_units": mdes(n, base_sd),
                "mdes_in_dB": mdes(n, base_sd) * base_sd,
            }

        R["splits"][split] = S

    # 闭环误差（论文 §5.3 的六个数量级）
    for f, key in (("verify_roundtrip.json", "initial"),
                   ("roundtrip_checkpoints_w3.json", "checkpoints_S1"),
                   ("roundtrip_checkpoints_lit.json", "checkpoints_S2")):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            R["roundtrip"][key] = load_json(p)

    # 干预实验
    p = os.path.join(HERE, "intervention_results.json")
    if os.path.exists(p):
        R["intervention"] = load_json(p)

    return R


# ------------------------------------------------------------ main
def main():
    R = analyse()

    print("=" * 78)
    print("扩展分析（独立单元 = 训练种子）")
    print("=" * 78)
    for split, S in R["splits"].items():
        print("\n### %s" % split)
        for arm, a in S["arms"].items():
            print("  %-14s n=%d  PSNR=%.4f±%.4f  (逐切片 p5=%.2f p95=%.2f, "
                  "种子内sd=%.3f 种子间sd=%.3f)"
                  % (LABELS[arm], a["n_seeds"], a["mean"], a["sd"],
                     a["slice_p05"], a["slice_p95"],
                     a["within_seed_sd"], a["across_seed_sd"]))
        print("  配对比较（独立单元=种子）:")
        for k, c in S["comparisons"].items():
            star = "***" if c["p"] < 0.001 else ("**" if c["p"] < 0.01 else
                                                 ("*" if c["p"] < 0.05 else "n.s."))
            mde = S["power"]["n=%d" % c["n"]]["mdes_in_dB"]
            ratio = abs(c["delta"]) / mde if mde > 0 else float("nan")
            print("    %-28s Δ=%+.4f  p=%.4f %-4s d_z=%+.2f  CI=[%+.3f,%+.3f]  "
                  "|Δ|/MDES=%.2f×"
                  % (k, c["delta"], c["p"], star, c["cohens_dz"],
                     c["ci"][0], c["ci"][1], ratio))
        vt = S.get("variance_test")
        if vt:
            print("  种子间方差齐性 (Levene): stat=%.3f p=%.3f  sd=%s"
                  % (vt["levene_stat"], vt["levene_p"],
                     ["%.4f" % v for v in vt["sds"]]))
        print("  功效（最小可检测效应, 80% power）:")
        for k, v in S["power"].items():
            print("    %-6s MDES = %.3f×sd = %.4f dB" % (k, v["mdes_in_sd_units"], v["mdes_in_dB"]))
        print("  收敛动力学:")
        for arm, cs in S.get("convergence", {}).items():
            print("    %-14s 末轮train_l1=%.6f  最优val_psnr=%.4f  最优轮次=%s  末段抖动=%.6f"
                  % (LABELS[arm], cs["final_train_l1"], cs["best_val_psnr"],
                     cs["best_epoch_each"], cs["late_epoch_instability"]))

    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=False)
    print("\n已写入 %s" % OUT_JSON)

    return R


if __name__ == "__main__":
    main()
