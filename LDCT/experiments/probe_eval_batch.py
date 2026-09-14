"""
定位：RED-CNN 在 512^2 评测时的显存/耗时是否随 batch 突变。

已测到的事实（probe_phase_mem.py）
------------------------------------------------------------------
  bs=1  512^2 裸前向：  0.68 s   峰值   491 MB
  bs=2  512^2 裸前向： 38.09 s   峰值  7047 MB     <-- 38 秒！
且 10 次评测步总共分配 669 GB。这不像容量问题，像某个库在按 batch
切换到一个 workspace 巨大的算法（cuDNN 对 ConvTranspose2d 的算法选择
依赖 shape，含 batch）。

本脚本把 batch 与「是否已训练过」两个因素分开：
  1. 全新模型下扫 bs ∈ {1,2,4}
  2. 训练 50 步后再扫一遍
  3. 关掉 cuDNN 再测 bs=2，验证是否 cuDNN 算法选择所致

用法
------------------------------------------------------------------
    python experiments/probe_eval_batch.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from models.redcnn import REDCNN          # noqa: E402


def mb(x):
    return x / 1024 ** 2


def run_case(model, bs, size, cudnn):
    torch.backends.cudnn.enabled = cudnn
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    x = torch.rand(bs, 1, size, size, device="cuda")
    with torch.no_grad():
        t0 = time.time()
        _ = model(x)
        torch.cuda.synchronize()
        dt = time.time() - t0
    peak = mb(torch.cuda.max_memory_allocated())
    torch.backends.cudnn.enabled = True
    del x
    tag = "cuDNN" if cudnn else "cuDNN关"
    print(f"  {tag:<7} bs={bs:<3} {size}^2    {dt:8.2f} s   峰值 {peak:8.1f} MB",
          flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    print(f"GPU {torch.cuda.get_device_name(0)}", flush=True)
    model = REDCNN().to("cuda")

    print("\n[1] 全新模型（只做前向，没训练过）", flush=True)
    model.eval()
    for bs in (1, 2, 4):
        run_case(model, bs, 512, True)

    print("\n[2] cuDNN 关闭时 bs=2（验证是否算法选择所致）", flush=True)
    run_case(model, 2, 512, False)

    print("\n[3] 训练 50 步之后重测", flush=True)
    model.train()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.L1Loss()
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(50):
        x = torch.rand(8, 1, 128, 128, device="cuda")
        y = torch.rand(8, 1, 128, 128, device="cuda")
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    model.eval()
    for bs in (1, 2, 4):
        run_case(model, bs, 512, True)

    print("\n判读：若 bs=1 快而 bs=2 慢，则评测批大小必须为 1；", flush=True)
    print("      若关掉 cuDNN 后 bs=2 变快，则根因是 cuDNN 的算法选择。", flush=True)


if __name__ == "__main__":
    main()
