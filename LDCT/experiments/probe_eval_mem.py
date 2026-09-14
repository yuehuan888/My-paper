"""
最小显存探针：RED-CNN 在 512×512 上，**评测**阶段到底吃掉多少显存、耗时多少。

为什么单独写这个
------------------------------------------------------------------
`diag_redcnn_mem.py` 里带评测的那一段，实测跑 40 张切片超过 11 分钟仍无输出
（显存 3783/4096 MiB、GPU 98%），说明它已经陷进 WDDM 换页，慢到无法给出
可用数字——**用一个已经饱和的测量去测饱和，是测不出 的**。

本脚本反过来：每次只测一小步，**每步之间 reset_peak + empty_cache**，
把峰值逐项拆开，找出真正把卡顶满的那一项。

用法
------------------------------------------------------------------
    python experiments/probe_eval_mem.py
    python experiments/probe_eval_mem.py --batch 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from models.redcnn import REDCNN                  # noqa: E402
from utils.metrics import ssim_torch              # noqa: E402


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def _safe_stdout() -> None:
    """Windows 控制台是 GBK，打印 '²' 之类字符会直接 UnicodeEncodeError 崩掉。
    本脚本只输出数字，故降到 ASCII-safe。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def reserved_mb() -> float:
    return torch.cuda.max_memory_reserved() / 1024 ** 2


def fresh(dev):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def report(name, t0, n=1):
    torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    print(f"  {name:<46} {dt*1000:8.1f} ms/张   "
          f"峰值 allocated {peak_mb():7.1f} MB / reserved {reserved_mb():7.1f} MB",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    _safe_stdout()
    dev = torch.device("cuda")
    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name}  {prop.total_memory/1024**2:.0f} MiB", flush=True)

    model = REDCNN().to(dev).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"RED-CNN 参数量 {n_par:,}\n", flush=True)

    x = torch.rand(args.batch, 1, args.size, args.size, device=dev)
    y = torch.rand(args.batch, 1, args.size, args.size, device=dev)

    # ---- 1. 参数 + 输入常驻显存 ----
    fresh(dev)
    _ = x * 1.0
    print(f"  常驻（参数+一个 batch 输入）峰值 {peak_mb():.1f} MB\n", flush=True)

    # ---- 2. 前向 ----
    with torch.no_grad():
        fresh(dev)
        t0 = time.time()
        out = model(x)
        report(f"前向 batch={args.batch} {args.size}^2", t0, args.batch)
        del out

    # ---- 3. 前向 + autocast(fp16) ----
    with torch.no_grad():
        fresh(dev)
        t0 = time.time()
        with torch.amp.autocast("cuda"):
            out = model(x)
        report(f"前向 batch={args.batch} {args.size}^2  autocast", t0, args.batch)
        del out

    # ---- 4. SSIM（口径 A）----
    with torch.no_grad():
        fresh(dev)
        t0 = time.time()
        s = ssim_torch(y, y)
        report(f"ssim_torch batch={args.batch} {args.size}^2", t0, args.batch)
        del s

    # ---- 5. 完整评测一步（前向 + PSNR + SSIM）----
    with torch.no_grad():
        fresh(dev)
        t0 = time.time()
        o = model(x)
        mse = (o - y).pow(2).flatten(1).mean(1)
        _ = 10.0 * torch.log10(400.0 ** 2 / mse.clamp_min(1e-12))
        _ = ssim_torch(o, y)
        report(f"完整评测一步 batch={args.batch} {args.size}^2", t0, args.batch)

    print("\n--- torch 分配器明细（最后一步后）---", flush=True)
    print(torch.cuda.memory_summary(abbreviated=True), flush=True)


if __name__ == "__main__":
    main()
