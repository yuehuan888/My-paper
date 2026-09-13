"""
低剂量 CT 去噪的训练 / 评测脚本。

设计原则（沿用 MGF-Net 项目的教训，见 `../MGF-Net_项目状态_2026-09-13.md` §四）
------------------------------------------------------------------
1. **固定随机种子**，保存完整配置（含数据来源、代码版本）
2. **按验证集选模**，不拿训练损失当 best
3. **测试集只评一次**
4. **必须跑平凡基线**：`identity`（输出=输入）给出噪声地板，
   `-` 模型若打不过它则毫无意义。MGF-Net 项目正是靠这一条
   发现了"零参数平均击败训练模型"。
5. **可复现**：每次运行的 config 落在输出目录

用法
------------------------------------------------------------------
    # 先看平凡基线（不需要训练）
    python experiments/train.py --baseline-only

    # 训练
    python experiments/train.py --tag pr_wavelet --epochs 30 --patch 128

    # 对照：不用 PR-LWT（旧式分离参数）
    python experiments/train.py --tag no_pr --no-pr-wavelet
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data.dataset import LoDoPaBDataset          # noqa: E402
from models.denoiser import PRWaveletDenoiser, count_parameters  # noqa: E402
from utils.metrics import psnr, ssim             # noqa: E402

DATA_ROOT = os.path.join(ROOT, "data", "extracted")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def code_version() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{rev[:12]}{'-dirty' if dirty else ''}"
    except Exception:  # noqa: BLE001
        return "unavailable"


# ------------------------------------------------------------------ 评测
@torch.no_grad()
def evaluate(model, loader, device, limit=None):
    """逐样本评测，返回逐样本指标与均值。"""
    model.eval()
    rows = []
    for i, (obs, gt) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        obs_d, gt_d = obs.to(device), gt.to(device)
        if model is None:
            out = obs_d                       # identity 基线
        else:
            out = model(obs_d)
        o = out.squeeze().float().cpu().numpy()
        g = gt_d.squeeze().float().cpu().numpy()
        rows.append({"PSNR": psnr(o, g), "SSIM": ssim(o, g)})
    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return rows, mean


def run_trivial_baselines(device, split="test", limit=None):
    """平凡基线：identity（输出=输入）。这是模型必须超过的地板。"""
    ds = LoDoPaBDataset(DATA_ROOT, split)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    print("=" * 78)
    print(f"平凡基线（{split} 集，{ds.n_slices} 张）")
    print("=" * 78)
    _, mean = evaluate(None, dl, device, limit=limit)
    print(f"  {'identity（输出=输入）':<26} PSNR={mean['PSNR']:.4f}  SSIM={mean['SSIM']:.4f}")
    print()
    print("  判读：模型若打不过 identity，说明它连'不做事'都不如。")
    return mean


# ------------------------------------------------------------------ 主流程
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(run_dir, exist_ok=True)

    tr = LoDoPaBDataset(DATA_ROOT, "train", patch_size=args.patch,
                        is_training=True)
    va = LoDoPaBDataset(DATA_ROOT, "validation", patch_size=None)
    te = LoDoPaBDataset(DATA_ROOT, "test", patch_size=None)

    model = PRWaveletDenoiser(levels=args.levels, mid_ch=args.mid_ch,
                              n_conv=args.n_conv,
                              use_pr_wavelet=not args.no_pr_wavelet,
                              global_residual=args.global_residual).to(device)
    n_par = count_parameters(model)
    rep = model.param_report()

    cfg = {
        "tag": args.tag, "created": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(), "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "patch": args.patch, "learning_rate": args.learning_rate,
        "lr_milestones": None, "levels": args.levels,
        "mid_ch": args.mid_ch, "n_conv": args.n_conv,
        "use_pr_wavelet": not args.no_pr_wavelet,
        "global_residual": args.global_residual,
        "n_params": n_par, "param_report": rep,
        "data_root": os.path.relpath(DATA_ROOT, ROOT),
        "n_train": tr.n_slices, "n_val": va.n_slices, "n_test": te.n_slices,
        "slice_shape": tr.slice_shape(),
        "device": str(device), "torch": torch.__version__,
    }

    milestones = [int(round(args.epochs * f)) for f in (0.5, 0.8, 0.9)]
    milestones = sorted(set(m for m in milestones if 0 < m < args.epochs))
    cfg["lr_milestones"] = milestones
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("=" * 78)
    print(f"训练 {args.tag}")
    print(f"  数据      : train={tr.n_slices}  val={va.n_slices}  test={te.n_slices}  "
          f"单片 {tr.slice_shape()}")
    print(f"  模型      : {n_par:,} 参数 (wavelet {rep['wavelet']}, heads {rep['heads']})"
          + ("" if not args.no_pr_wavelet else "   [旧式非 PR 变换]"))
    print(f"  配置      : epochs={args.epochs} patch={args.patch} bs={args.batch_size} "
          f"lr={args.learning_rate} LR衰减点={milestones}")
    print(f"  代码版本  : {cfg['code_version']}")
    print("=" * 78)

    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    va_loader = DataLoader(va, batch_size=1, shuffle=False, num_workers=0)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                               gamma=0.5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"epoch": [], "train_l1": [], "val_psnr": [], "val_ssim": []}
    best = {"psnr": -np.inf, "epoch": None, "state": None, "val": None}
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for obs, gt in tr_loader:
            obs, gt = obs.to(device), gt.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    out = model(obs)
                    loss = criterion(out, gt)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(obs)
                loss = criterion(out, gt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        train_l1 = float(np.mean(losses))
        history["epoch"].append(ep)
        history["train_l1"].append(train_l1)

        if ep % args.val_interval == 0 or ep == args.epochs:
            _, vm = evaluate(model, va_loader, device, limit=args.val_limit)
            history["val_psnr"].append(vm["PSNR"])
            history["val_ssim"].append(vm["SSIM"])
            tag = ""
            if vm["PSNR"] > best["psnr"]:
                best.update(psnr=vm["PSNR"], epoch=ep, val=vm,
                            state={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})
                tag = "  *best*"
            print(f"  epoch {ep:3d} | train L1 {train_l1:.5f} | "
                  f"val PSNR {vm['PSNR']:.3f}  SSIM {vm['SSIM']:.4f}{tag}", flush=True)
        else:
            print(f"  epoch {ep:3d} | train L1 {train_l1:.5f}", flush=True)

    wall = time.time() - t0

    # 用验证集选出的权重评测测试集
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    te_loader = DataLoader(te, batch_size=1, shuffle=False, num_workers=0)
    te_rows, te_mean = evaluate(model, te_loader, device, limit=args.test_limit)

    results = {
        "tag": args.tag, "seed": args.seed, "n_params": n_par,
        "best_epoch": best["epoch"], "best_val_psnr": best["psnr"],
        "val_mean": best["val"], "test_mean": te_mean,
        "test_per_sample": te_rows,
        "wall_seconds": wall,
    }
    ckpt = os.path.join(run_dir, "best.pth")
    torch.save({"epoch": best["epoch"], "model_state_dict": model.state_dict(),
                "config": cfg}, ckpt)
    results["checkpoint"] = os.path.relpath(ckpt, ROOT)
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("-" * 78)
    print(f"最优 epoch {best['epoch']}  (val PSNR {best['psnr']:.4f})   用时 {wall/60:.1f} min")
    print(f"TEST:  PSNR {te_mean['PSNR']:.4f}   SSIM {te_mean['SSIM']:.4f}   "
          f"(n={len(te_rows)})")
    print(f"结果目录: {run_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description="LDCT 去噪训练")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--baseline-only", action="store_true",
                    help="只跑平凡基线（identity），不训练")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--mid-ch", type=int, default=16)
    ap.add_argument("--n-conv", type=int, default=2)
    ap.add_argument("--no-pr-wavelet", action="store_true",
                    help="用旧式分离参数变换作对照")
    ap.add_argument("--global-residual", action="store_true")
    ap.add_argument("--val-interval", type=int, default=2)
    ap.add_argument("--val-limit", type=int, default=None,
                    help="验证时只评前 N 张（加速）")
    ap.add_argument("--test-limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline_only:
        run_trivial_baselines(device, "test", args.test_limit)
        return
    if not args.tag:
        ap.error("非 --baseline-only 时必须给 --tag")
    train(args)


if __name__ == "__main__":
    main()
