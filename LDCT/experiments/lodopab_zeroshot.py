"""
LoDoPaB-CT 零样本外部测试：用 AAPM 上训好的权重，**不做任何微调**直接跑另一个库。

为什么值得做
------------------------------------------------------------------
投稿路线报告把它列为"拒稿理由 #2（单数据集、无跨库泛化）"的应对：
哪怕掉点，能证明泛化性就把"结构性弱点"变成"加分项"。

口径必须先说清楚（否则数字没意义）
------------------------------------------------------------------
`utils/metrics.py` 的 `ssim_torch` 内部**硬编码了 AAPM 的 HU 变换**
（`x*4096-1024` 再 clip 到 [-160,240]，data_range=400）。
LoDoPaB 的 [0,1] 是它自己的缩放，**不能套用那套口径**。

故本脚本自带一套原生指标（`[0,1]` 域 + 可配 data_range），
结构沿用同一套 SSIM 实现（零填充 11×11 高斯、全图均值），只去掉 HU 变换。

**真正的风险是"尺度错配"而不是指标**：若 LoDoPaB 的 [0,1] 与
AAPM 的 `(HU+1024)/4096` 不是同一套缩放，直接喂进去就是分布外输入，
数字会很难看但**不会报错**。故本脚本先做分布诊断，再决定是否需要对齐。

用法
------------------------------------------------------------------
    python experiments/lodopab_zeroshot.py --inspect
    python experiments/lodopab_zeroshot.py --limit 400
    python experiments/lodopab_zeroshot.py --limit 400 --align
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from data.dataset import LoDoPaBDataset        # noqa: E402
from models.denoiser import PRWaveletDenoiser  # noqa: E402
from utils.metrics import _ssim_gauss1d        # noqa: E402

LODO_ROOT = os.path.join(ROOT, "data", "extracted")
RUNS = os.path.join(HERE, "runs")

# 口径 A 在 AAPM 上把 HU 映射到 [0,1]：HU=-1024 → 0
AAPM_HU_OFFSET, AAPM_HU_SCALE = 1024.0, 4096.0


# ---------------------------------------------------------------- 原生指标
def psnr_01(pred: torch.Tensor, target: torch.Tensor,
            data_range: float = 1.0) -> torch.Tensor:
    mse = (pred - target).pow(2).flatten(1).mean(1)
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))


def ssim_01(pred: torch.Tensor, target: torch.Tensor,
            data_range: float = 1.0) -> torch.Tensor:
    """与 utils.metrics.ssim_torch 同结构，但**不做 HU 变换**、不 clip。"""
    g1 = _ssim_gauss1d()
    w = torch.as_tensor(np.outer(g1, g1), dtype=pred.dtype,
                        device=pred.device).view(1, 1, len(g1), len(g1))
    p, g = pred.float(), target.float()
    mu1 = F.conv2d(g, w, padding=5)
    mu2 = F.conv2d(p, w, padding=5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(g * g, w, padding=5) - mu1_sq
    s2 = F.conv2d(p * p, w, padding=5) - mu2_sq
    s12 = F.conv2d(g * p, w, padding=5) - mu1_mu2
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return m.flatten(1).mean(1)


# ---------------------------------------------------------------- 分布诊断
@torch.no_grad()
def inspect(loader, device, n=8):
    """看清 LoDoPaB 的尺度，并与 AAPM 的 (HU+1024)/4096 对比。"""
    obs, gt = [], []
    for i, (o, g) in enumerate(loader):
        if i >= n:
            break
        obs.append(o)
        gt.append(g)
    o = torch.cat(obs).float()
    g = torch.cat(gt).float()

    def desc(name, t):
        q = torch.quantile(t.flatten(), torch.tensor(
            [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0], device=t.device))
        print(f"  {name:<14} shape={tuple(t.shape)}  mean={t.mean():+.4f} "
              f"std={t.std():.4f}")
        print(f"    min/0.1%/1%/中位/99%/99.9%/max = "
              + "  ".join(f"{v:+.4f}" for v in q.tolist()))
        print(f"    负值 {float((t < 0).float().mean())*100:.4f}%   "
              f">1 占比 {float((t > 1).float().mean())*100:.4f}%")

    print("\n--- LoDoPaB test ---")
    desc("observation", o)
    desc("ground_truth", g)
    d = o - g
    print(f"  obs-gt 差: mean={d.mean():+.5f} std={d.std():.5f}  "
          f"（噪声强度）")
    # 只差一个仿射？最小二乘拟合 obs ≈ a*gt + b
    x = g.flatten().to(torch.float64)
    y = o.flatten().to(torch.float64)
    A = torch.stack([x, torch.ones_like(x)], 1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze()
    a, b = float(sol[0]), float(sol[1])
    resid = float((A @ sol - y.unsqueeze(1)).abs().mean())
    print(f"  线性拟合 obs ≈ {a:.6f}·gt {b:+.6f}   残差均值 {resid:.6f}")

    print("\n--- 参照：AAPM 的 (HU+1024)/4096 ---")
    print("  实测（L067，见 lodopab_inspect.py 输出）：")
    print("    gt  min/中位/99%/max = +0.0000 / +0.2209 / +0.3438 / +0.6641")
    print("    若 LoDoPaB 的中位数与 0.22 相差很远，说明两套 [0,1] 不是同一缩放。")
    return o, g


# ---------------------------------------------------------------- 模型
def load_model(tag, device):
    ck = os.path.join(RUNS, tag, "best.pth")
    if not os.path.exists(ck):
        return None
    st = torch.load(ck, map_location=device, weights_only=False)
    kind = st["config"].get("wavelet", "pr")
    m = PRWaveletDenoiser(levels=st["config"].get("levels", 2),
                          mid_ch=st["config"].get("mid_ch", 16),
                          n_conv=st["config"].get("n_conv", 2),
                          wavelet=kind).to(device)
    m.load_state_dict(st["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def run_eval(model, loader, device, limit, data_range, affine=None):
    """affine=(a,b) 时，先把输入映射到模型训练域，再把输出映射回来。"""
    rows = []
    for i, (o, g) in enumerate(loader):
        if i >= limit:
            break
        o, g = o.to(device).float(), g.to(device).float()
        x = o if affine is None else (o - affine[1]) / affine[0]
        if model is None:
            out = x
        else:
            out = model(x)
        if affine is not None:
            out = out * affine[0] + affine[1]
        rows.append((float(psnr_01(out, g, data_range).mean()),
                     float(ssim_01(out, g, data_range).mean())))
    ps = np.array([r[0] for r in rows])
    ss = np.array([r[1] for r in rows])
    return float(ps.mean()), float(ss.mean()), len(rows)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="评测切片数")
    ap.add_argument("--data-range", type=float, default=1.0)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--align", action="store_true",
                    help="启用尺度对齐（把 LoDoPaB 仿射映射到 AAPM 域再映射回来）")
    ap.add_argument("--tags", default=None,
                    help="逗号分隔的 run 名；默认用全部 w3_pr_*/lit_pr_*")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LoDoPaBDataset(LODO_ROOT, "test", patch_size=None, is_training=False)
    print(ds.summary(), flush=True)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    if args.inspect:
        inspect(loader, device)
        return

    # ---- 尺度对齐系数：把 LoDoPaB 对齐到 AAPM 的域 ----
    affine = None
    if args.align:
        o, g = inspect(loader, device, n=8)
        # 用 gt 的均值/标准差做线性匹配（在 [0,1] 域内）
        sg = g.std()
        # AAPM gt: mean 0.1478, std 0.1228（实测，见 lodopab_inspect.py）
        A_MEAN, A_STD = 0.1478, 0.1228
        a = float(sg / A_STD)
        b = float(g.mean() - a * A_MEAN)
        affine = (a, b)
        print(f"\n  [align] 对齐系数 a={a:.6f} b={b:+.6f}"
              f"（LoDoPaB gt → AAPM 域）", flush=True)

    tags = (args.tags.split(",") if args.tags else
            [f"w3_pr_s{i}" for i in range(5)] +
            [f"lit_pr_s{i}" for i in range(3)])

    print(f"\n{'=' * 78}\n零样本评测：{len(tags)} 个 AAPM 权重，"
          f"LoDoPaB test 前 {args.limit} 张，data_range={args.data_range}"
          f"{'，尺度对齐已启用' if affine else ''}\n{'=' * 78}", flush=True)
    print(f"  {'模型':<16}{'PSNR':>9}{'SSIM':>9}{'n':>6}{'耗时':>9}", flush=True)
    print("  " + "-" * 52, flush=True)

    results = {}

    t0 = time.time()
    p, s, n = run_eval(None, loader, device, args.limit, args.data_range, affine)
    print(f"  {'identity（地板）':<16}{p:>9.4f}{s:>9.4f}{n:>6}"
          f"{time.time()-t0:>8.0f}s", flush=True)
    results["identity"] = {"PSNR": p, "SSIM": s, "n": n}

    for tag in tags:
        m = load_model(tag, device)
        if m is None:
            print(f"  {tag:<16}  (无 checkpoint)", flush=True)
            continue
        t0 = time.time()
        p, s, n = run_eval(m, loader, device, args.limit, args.data_range, affine)
        print(f"  {tag:<16}{p:>9.4f}{s:>9.4f}{n:>6}{time.time()-t0:>8.0f}s",
              flush=True)
        results[tag] = {"PSNR": p, "SSIM": s, "n": n}

    print("  " + "-" * 52, flush=True)
    print("\n判读：与 identity 比。若模型打不过 identity，说明权重没能跨库迁移。",
          flush=True)

    dst = os.path.join(HERE, "lodopab_zeroshot_results.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump({"limit": args.limit, "data_range": args.data_range,
                   "align": affine, "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写入 {dst}", flush=True)


if __name__ == "__main__":
    main()
