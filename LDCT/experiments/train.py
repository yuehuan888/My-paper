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

from data.dataset import LoDoPaBDataset, AapmDataset  # noqa: E402
from models.denoiser import PRWaveletDenoiser, count_parameters  # noqa: E402
from utils.metrics import (psnr, ssim, ssim_torch,       # noqa: E402
                           HU_OFFSET, HU_SCALE, EVAL_LO, EVAL_HI, DATA_RANGE)

DATA_ROOT = os.path.join(ROOT, "data", "extracted")
AAPM_ROOT = os.path.join(ROOT, "data", "aapm_h5")
AAPM_SPLIT = os.path.join(ROOT, "splits", "aapm_mayo_3mm.json")


def make_dataset(name, split, split_file=None, **kw):
    """统一入口。AAPM 用按患者的划分；LoDoPaB 用官方 split。"""
    if name == "aapm":
        return AapmDataset(AAPM_ROOT, split_file or AAPM_SPLIT, split, **kw)
    return LoDoPaBDataset(DATA_ROOT, split, **kw)


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


# ------------------------------------------------------- 落盘与日志（可恢复性）
#
# 背景（2026-09-14）：RED-CNN 基线的 5 次运行（redcnn_s0/s1/s2、lit_redcnn_s0/s1）
# 全部只剩下一个 config.json——因为旧版 train.py **只在 30 轮全部跑完的那一刻**
# 才写 history.json / results.json，中途任何中断都等于从零开始。
# 且 stdout 无落盘，进程一旦脱离终端即成黑盒：实测有进程跑了 33 分钟，
# 期间**没有写过任何一个文件**，无法判断它在第几轮。
#
# 以下三件事修的就是这个：
#   1. `_Tee`：stdout 同时进 runs/<tag>/train.log
#   2. `_atomic_json`：先写 .tmp 再 os.replace，断电也不会留下半个文件
#   3. 每轮落盘 progress.json / history.json / last.pth，并支持 --resume
class _Tee:
    """把 stdout 复制一份到日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:  # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:  # noqa: BLE001
                pass

    def isatty(self):
        return False


def _atomic_json(path: str, obj) -> None:
    """原子写 JSON：先写同目录 .tmp，再 os.replace 覆盖。

    直接 open(path,'w') 写到一半断电/被杀，会留下截断的 JSON，
    下次读取直接抛异常——本项目已经吃过一次。
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _install_log(run_dir: str):
    """把 stdout 接到 run_dir/train.log，返回 (原始 stdout, 文件句柄)。"""
    fh = open(os.path.join(run_dir, "train.log"), "a",
              encoding="utf-8", buffering=1)
    real = sys.stdout
    sys.stdout = _Tee(real, fh)
    return real, fh


# ------------------------------------------------------------------ 评测
@torch.no_grad()
def evaluate(model, loader, device, limit=None, batch_eval=None):
    """评测，返回逐样本指标与均值。**在 GPU 上整批完成**。

    为什么必须整批：口径 A 的评测要跑 343~435 张 512×512 切片，
    CPU 版 SSIM 约 279 ms/张，一次验证近 1 分钟——30 轮训练里验证占总时长
    约 70%（训练本身只要 2.5 分钟）。GPU 版 SSIM 快 **12 倍**且数值等价
    （最大差 1.07e-07，见 utils/metrics.py 的 `ssim_torch`）。
    """
    if model is not None:
        model.eval()          # identity 基线时 model 为 None

    # 评估批大小必须随模型规格调整。**下面是实测值，不是估计值**
    # （experiments/probe_eval_batch.py，RTX 3050 Laptop 4GB，512×512 前向）：
    #
    #   RED-CNN (96 通道 + ConvTranspose2d)   bs=1 →  0.70 s /  489 MB
    #                                         bs=2 → 34.17 s / 7025 MB
    #                                         bs=4 → 41.26 s / 7603 MB
    #   RED-CNN + cuDNN 关闭                   bs=2 →  6.87 s / 2994 MB
    #
    # 即 **batch 从 1 涨到 2，耗时跳 49 倍、显存跳 14 倍**，且关掉 cuDNN 后同一
    # batch 快 5 倍 —— 根因是 cuDNN 为 RED-CNN 的 ConvTranspose2d 在这个 shape
    # 上挑了一个 workspace 约 7 GB 的算法。4GB 的卡装不下，溢出到 WDDM 共享
    # 内存，此后每次访问都走 PCIe。
    #
    # 代价有多大：一次验证 343 张，bs=2 要 172×34 s ≈ **97 分钟**，
    # 六次验证 + 测试约 **12 小时**——这正是上一轮"跑了 33 分钟没有任何输出"
    # 的真正原因，不是卡死，是在以 1/50 的速度爬。
    #
    # 批大小**不影响数值**：evaluate() 逐切片算 PSNR，ssim_torch 也是逐样本
    # 返回（m.flatten(1).mean(1)），零填充在样本内完成。故 bs=1 与 bs=2 结果
    # 逐位相同，只差速度。
    if batch_eval is None:
        n_par = sum(p.numel() for p in model.parameters()) if model is not None else 0
        batch_eval = 16 if n_par < 100_000 else 1

    obs_list, gt_list = [], []
    for i, (obs, gt) in enumerate(loader):
        if limit is not None and i >= limit:
            break
        obs_list.append(obs)
        gt_list.append(gt)
    if not obs_list:
        raise ValueError("评测集为空")

    rows = []
    for s in range(0, len(obs_list), batch_eval):
        ob = torch.cat(obs_list[s:s + batch_eval], 0).to(device).float()
        gt = torch.cat(gt_list[s:s + batch_eval], 0).to(device).float()
        out = ob if model is None else model(ob)

        # PSNR / SSIM 均在 GPU 上按口径 A 计算
        p = torch.clamp(out * HU_SCALE - HU_OFFSET, EVAL_LO, EVAL_HI)
        g = torch.clamp(gt * HU_SCALE - HU_OFFSET, EVAL_LO, EVAL_HI)
        mse = (p - g).pow(2).flatten(1).mean(1)
        ps = 10.0 * torch.log10(DATA_RANGE ** 2 / mse.clamp_min(1e-12))
        ps = torch.where(mse <= 0, torch.full_like(ps, float("inf")), ps)
        ss = ssim_torch(out, gt)
        for j in range(ob.shape[0]):
            rows.append({"PSNR": float(ps[j]), "SSIM": float(ss[j])})

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return rows, mean


def run_trivial_baselines(device, split="test", limit=None, dataset="aapm",
                          split_file=None):
    """平凡基线：identity（输出=输入）。这是模型必须超过的地板。"""
    ds = make_dataset(dataset, split, split_file)
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

    real_stdout, log_fh = _install_log(run_dir)
    print(f"\n{'=' * 78}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"tag={args.tag}  日志 -> {os.path.join(run_dir, 'train.log')}")

    # 训练集载入内存：单线程逐样本开 gzip HDF5 极慢（实测一轮 30 分钟）
    tr = make_dataset(args.dataset, "train", split_file=args.split_file,
                      patch_size=args.patch, is_training=True, cache=True)
    va = make_dataset(args.dataset, "val", split_file=args.split_file,
                      patch_size=None, cache=True)
    te = make_dataset(args.dataset, "test", split_file=args.split_file,
                      patch_size=None, cache=True)

    if args.model == "redcnn":
        from models.redcnn import REDCNN
        model = REDCNN(in_channels=1, out_channels=1).to(device)
    else:
        model = PRWaveletDenoiser(levels=args.levels, mid_ch=args.mid_ch,
                                  n_conv=args.n_conv,
                                  wavelet=args.wavelet,
                                  global_residual=args.global_residual,
                                  synth_mismatch=args.synth_mismatch,
                                  n_taps=args.n_taps,
                                  bound=args.bound,
                                  init_drift=args.init_drift).to(device)
    n_par = count_parameters(model)
    # 只有 PRWaveletDenoiser 提供参数构成与闭环诊断；RED-CNN 无此接口
    rep = (model.param_report() if hasattr(model, "param_report")
           else {"wavelet": 0, "heads": n_par, "global": 0,
                 "total": n_par, "wavelet_frac": 0.0})

    cfg = {
        "tag": args.tag, "model": args.model,
        "created": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(), "seed": args.seed,
        "epochs": args.epochs, "batch_size": args.batch_size,
        # ⚠️ 更正（2026-09-16）：此处原写"parallelism 影响速度但不影响数值"——**是错的**。
        #    DataLoader 的每个 worker 进程有独立的 torch RNG 状态，而随机裁块正是
        #    从 torch 全局 RNG 取的。实测同 seed=0：workers=0 得 30.8684、
        #    workers=4 得 30.8172（差 0.051 dB）。换 workers 等于换了一次实验。
        "workers": args.workers,
        "patch": args.patch, "learning_rate": args.learning_rate,
        "lr_milestones": None, "levels": args.levels,
        "mid_ch": args.mid_ch, "n_conv": args.n_conv,
        # 提升格式旋钮。默认 3 / 0.5 = 全部已有结果所用的配置；
        # 不记录的话 bound 扫描的结果无法区分是哪个 bound 跑出来的。
        "n_taps": args.n_taps, "bound": args.bound,
        "init_drift": args.init_drift,
        "wavelet": args.wavelet,
        "global_residual": args.global_residual,
        # 干预实验的注入强度。非 0 时本次运行是"被注入闭环误差的 PR 臂"，
        # 结果不能与常规 pr 臂混为一谈——不记录就复原不出来。
        "synth_mismatch": args.synth_mismatch,
        "n_params": n_par, "param_report": rep,
        "dataset": args.dataset,
        "data_root": os.path.relpath(AAPM_ROOT if args.dataset=="aapm" else DATA_ROOT, ROOT),
        "split_file": args.split_file or AAPM_SPLIT,
        "n_train": tr.n_slices, "n_val": va.n_slices, "n_test": te.n_slices,
        "slice_shape": tr.slice_shape(),
        # 选模协议必须记录：val_interval 决定验证跑几次，直接决定总时长与
        # "最优轮"的可比性。旧版漏记，导致事后无法复原两次运行是否同协议。
        "val_interval": args.val_interval, "val_limit": args.val_limit,
        "test_limit": args.test_limit,
        "device": str(device), "torch": torch.__version__,
    }

    milestones = [int(round(args.epochs * f)) for f in (0.5, 0.8, 0.9)]
    milestones = sorted(set(m for m in milestones if 0 < m < args.epochs))
    cfg["lr_milestones"] = milestones
    _atomic_json(os.path.join(run_dir, "config.json"), cfg)

    print("=" * 78)
    print(f"训练 {args.tag}")
    print(f"  数据      : train={tr.n_slices}  val={va.n_slices}  test={te.n_slices}  "
          f"单片 {tr.slice_shape()}")
    print(f"  模型      : {args.model}  {n_par:,} 参数"
          + (f" (wavelet {rep['wavelet']}, heads {rep['heads']}, "
             f"wavelet={args.wavelet})" if args.model == "pr_wavelet" else ""))
    if hasattr(model, "transform_roundtrip"):
        me, rl = model.transform_roundtrip(torch.rand(1,1,256,256).to(device))
        print(f"  变换闭环  : max|err|={me:.3e}  rel_L2={rl:.3e}")
    else:
        print(f"  变换闭环  : 不适用（RED-CNN 无小波变换）")
    print(f"  配置      : epochs={args.epochs} patch={args.patch} bs={args.batch_size} "
          f"lr={args.learning_rate} LR衰减点={milestones}")
    print(f"  代码版本  : {cfg['code_version']}")
    print("=" * 78)

    # ---------------------------------------------------------------- 数据加载
    # 两条**实测**结论（2026-09-16，RTX 3050 6GB；模型仅 2,159 参数、激活 ~15 MB）：
    #
    # 1. `num_workers>0` **更慢**，不是更快。
    #    实测 30 轮：workers=0 → 162.9s；workers=4 → 177.6s。
    #    Windows 下 worker>0 走 spawn，进程启动 + 数据序列化开销超过并行收益，
    #    而 h5 已在内存缓存、读取本来就快。
    #    （曾误以为 GPU 空转是数据加载造成的 —— 错。真因见 run_parallel.py 的注释：
    #      是 kernel launch 开销，靠**并发跑多个进程**解决，不是靠 worker。）
    #
    # 2. `num_workers` **会改变数值**（不只是速度）。
    #    DataLoader 的每个 worker 进程有**独立的 torch RNG 状态**，而
    #    `AapmDataset.__getitem__` 的随机裁块正是从 torch 全局 RNG 取的
    #    （见 data/dataset.py 中 2026-09-16 的修复说明）。
    #    实测同 seed=0：workers=0 → 30.8684，workers=4 → 30.8172（差 0.051 dB）。
    #    ⚠️ 所以「换 workers 重跑」得到的是**不同的一次实验**，不能与旧数字并列。
    #
    # 结论：默认 `--workers 0`。要提速请用 experiments/run_parallel.py 并发跑多作业。
    _workers = max(0, int(args.workers))
    _persist = _workers > 0          # 无 worker 时 persistent 无意义
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=_workers, pin_memory=True, drop_last=True,
                           persistent_workers=_persist,
                           prefetch_factor=4 if _workers > 0 else None)
    va_loader = DataLoader(va, batch_size=1, shuffle=False, num_workers=0)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                               gamma=0.5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"epoch": [], "train_l1": [], "val_psnr": [], "val_ssim": []}
    best = {"psnr": -np.inf, "epoch": None, "state": None, "val": None}
    start_ep = 1

    # ---- 断点续训：本机有反复杀进程的历史，没有这个功能就会一次次从零开始 ----
    last_ckpt = os.path.join(run_dir, "last.pth")
    if args.resume:
        if not os.path.exists(last_ckpt):
            print(f"  [resume] 没有 {last_ckpt}，从头开始")
        else:
            st = torch.load(last_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(st["model"])
            optimizer.load_state_dict(st["optimizer"])
            scheduler.load_state_dict(st["scheduler"])
            if scaler is not None and st.get("scaler"):
                scaler.load_state_dict(st["scaler"])
            # MultiStepLR 的 milestones **不进 state_dict**（它是构造参数），
            # 故换 --epochs 续训会静默改变学习率轨迹。这里显式拦一道。
            old_ms = st.get("milestones")
            if old_ms is not None and list(old_ms) != list(milestones):
                print(f"  [resume] ⚠️  LR 衰减点不一致：存档 {list(old_ms)} "
                      f"vs 本次 {milestones}。"
                      f"（MultiStepLR 的 milestones 不进 state_dict）"
                      f"续训请使用与原 run 相同的 --epochs。")
            history, best = st["history"], st["best"]
            start_ep = int(st["epoch"]) + 1
            bp = best["psnr"]
            print(f"  [resume] 已完成 {st['epoch']}/{args.epochs} 轮，"
                  f"从 epoch {start_ep} 继续"
                  + (f"（最好 val PSNR {bp:.4f} @ ep{best['epoch']}）"
                     if best["epoch"] else ""))

    t0 = time.time()

    for ep in range(start_ep, args.epochs + 1):
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

        # ---------------- 每轮落盘（本次修复的核心）----------------
        # 旧版只在 30 轮全部跑完的那一刻才写结果，中途被杀 = 全部丢失。
        # 实测已发生 5 次（redcnn_s0/s1/s2、lit_redcnn_s0/s1）。
        _atomic_json(os.path.join(run_dir, "history.json"), history)
        _atomic_json(os.path.join(run_dir, "progress.json"), {
            "tag": args.tag, "epoch": ep, "epochs": args.epochs,
            "best_epoch": best["epoch"],
            "best_val_psnr": None if best["epoch"] is None else best["psnr"],
            "elapsed_min": (time.time() - t0) / 60,
            # 显存峰值：PR-LWT 实测 104 MB、RED-CNN 训练实测 104 MB、
            # 512² 评测峰值 491 MB（见 experiments/probe_eval_mem.py）。
            # 记下来是为了让"显存是不是瓶颈"变成可查的数字，而不是事后猜。
            "peak_vram_mb": (round(torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
                             if device.type == "cuda" else None),
            "updated": datetime.now(timezone.utc).isoformat(),
        })
        ck_tmp = last_ckpt + ".tmp"
        torch.save({"epoch": ep,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "milestones": milestones,
                    "history": history, "best": best}, ck_tmp)
        os.replace(ck_tmp, last_ckpt)

    wall = time.time() - t0

    # 用验证集选出的权重评测测试集
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    te_loader = DataLoader(te, batch_size=1, shuffle=False, num_workers=0)
    te_rows, te_mean = evaluate(model, te_loader, device, limit=args.test_limit)

    # 逐患者拆分：口径 A 的文献常用 **L506 单患者**做测试，
    # 故同时报 L506-only（可对外比）与全部测试患者（自有估计）
    per_patient = {}
    if hasattr(te, "_index"):
        from collections import defaultdict
        buckets = defaultdict(list)
        for row, (pid, _) in zip(te_rows, te._index):
            buckets[pid].append(row)
        for pid, rs in buckets.items():
            per_patient[pid] = {
                "n": len(rs),
                "PSNR": float(np.mean([r["PSNR"] for r in rs])),
                "SSIM": float(np.mean([r["SSIM"] for r in rs])),
            }

    results = {
        "tag": args.tag, "seed": args.seed, "n_params": n_par,
        "test_per_patient": per_patient,
        "best_epoch": best["epoch"], "best_val_psnr": best["psnr"],
        "val_mean": best["val"], "test_mean": te_mean,
        "test_per_sample": te_rows,
        "wall_seconds": wall,
    }
    ckpt = os.path.join(run_dir, "best.pth")
    torch.save({"epoch": best["epoch"], "model_state_dict": model.state_dict(),
                "config": cfg}, ckpt)
    results["checkpoint"] = os.path.relpath(ckpt, ROOT)
    _atomic_json(os.path.join(run_dir, "results.json"), results)
    _atomic_json(os.path.join(run_dir, "history.json"), history)

    print("-" * 78)
    print(f"最优 epoch {best['epoch']}  (val PSNR {best['psnr']:.4f})   用时 {wall/60:.1f} min")
    print(f"TEST:  PSNR {te_mean['PSNR']:.4f}   SSIM {te_mean['SSIM']:.4f}   "
          f"(n={len(te_rows)})")
    for pid, v in sorted(per_patient.items()):
        print(f"   {pid}: PSNR {v['PSNR']:.4f}  SSIM {v['SSIM']:.4f}  (n={v['n']})")
    print(f"结果目录: {run_dir}")

    # 完成标记：一行命令就能回答"这个 run 到底跑完了没"，
    # 不必再去比对 config.json / history.json 哪个存在。
    _atomic_json(os.path.join(run_dir, "DONE"), {
        "tag": args.tag, "epoch": best["epoch"], "epochs": args.epochs,
        "test_mean": te_mean, "wall_min": wall / 60,
        "finished": datetime.now(timezone.utc).isoformat(),
    })
    print(f"✅ 完成标记: {os.path.join(run_dir, 'DONE')}")
    print("=" * 78)

    sys.stdout = real_stdout          # 还原 stdout，否则日志句柄会一直挂着
    log_fh.close()
    return results


def main():
    ap = argparse.ArgumentParser(description="LDCT 去噪训练")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dataset", default="aapm", choices=["aapm", "lodopab"])
    ap.add_argument("--split-file", default=None,
                    help="划分文件；默认 splits/aapm_mayo_3mm.json。"
                         "splits/aapm_mayo_3mm_lit.json 为对齐文献的 L506 单测试划分")
    ap.add_argument("--baseline-only", action="store_true",
                    help="只跑平凡基线（identity），不训练")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    # 数据加载并行度。**默认 0（保持原行为）**。
    # 2026-09-16 实测：workers=4 反而更慢（177.6s vs 162.9s，30 轮）——
    # Windows 下 worker>0 走 spawn，进程启动与数据序列化的开销超过了并行收益，
    # 而 h5 读取本来就快。所以 GPU 空转**不是**数据加载造成的。
    # 真正的瓶颈是 kernel launch 开销（模型只有 2,159 参数，算子极小，
    # batch 8 / patch 128 每次 kernel 干的活不够填启动开销）。
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader worker 数（0=串行；实测 Windows 下 >0 更慢）")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--levels", type=int, default=2)
    # 提升格式的两个旋钮。默认值即全部已有结果所用的配置。
    # 暴露它们的用途：贡献 3（"结构 PR 在 float32 下不自动成立，必须界定 taps"）
    # 此前只有一个观测点（无界 → 5.2e+00），无法画剂量-响应曲线。
    ap.add_argument("--n-taps", type=int, default=3,
                    help="提升滤波器 P/U 的抽头数，必须为奇数（默认 3）")
    ap.add_argument("--init-drift", type=float, default=0.0,
                    help="把 taps 初始化在离 Haar 距离 d 处（E3 用）。"
                         "必须 < --bound，故 E3 需配 --bound 8.0。默认 0 = 严格 Haar。")
    ap.add_argument("--bound", type=float, default=0.5,
                    help="taps 相对 Haar 初始值的最大偏离，即 |taps| <= init + bound"
                         "（默认 0.5）。调大即模拟无界化，用于验证数值稳定性边界。")
    ap.add_argument("--mid-ch", type=int, default=16)
    ap.add_argument("--n-conv", type=int, default=2)
    ap.add_argument("--model", default="pr_wavelet",
                    choices=["pr_wavelet", "redcnn"],
                    help="redcnn = RED-CNN 基线（Chen 2017，约 1.85M 参数），"
                         "用于在同划分下与本文方法对比")
    ap.add_argument("--wavelet", default="pr",
                    choices=["pr", "fixed", "unconstrained"],
                    help="pr=PR-LWT(可逆,可学习) | fixed=固定Haar | "
                         "unconstrained=旧式分离参数(可学习但无闭环保证)。"
                         "三臂分别回答不同问题，勿混。")
    ap.add_argument("--global-residual", action="store_true")
    ap.add_argument("--synth-mismatch", type=float, default=0.0,
                    help="干预实验：向 PR 臂注入受控闭环误差。合成用 (1-ε)·U，"
                         "分解仍用 U。ε=0 时行为与普通 PR-LWT 完全一致。"
                         "标定见 experiments/calibrate_mismatch.py")
    ap.add_argument("--resume", action="store_true",
                    help="从 runs/<tag>/last.pth 继续训练（每轮自动保存）")
    ap.add_argument("--val-interval", type=int, default=2)
    ap.add_argument("--val-limit", type=int, default=None,
                    help="验证时只评前 N 张（加速）")
    ap.add_argument("--test-limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.baseline_only:
        run_trivial_baselines(device, "test", args.test_limit, args.dataset,
                              args.split_file)
        return
    if not args.tag:
        ap.error("非 --baseline-only 时必须给 --tag")
    train(args)


if __name__ == "__main__":
    main()
