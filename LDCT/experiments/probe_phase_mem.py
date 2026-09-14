"""
逐阶段累积显存探针：找出 RED-CNN 训练里到底是哪一步把分配器撑到 7 GB。

背景（2026-09-14 实测）
------------------------------------------------------------------
`runs/smoke_redcnn2/progress.json` 里 `peak_vram_mb = 7056.0` ——
在一张 **4096 MiB** 的卡上。这说明分配器拿了 7 GB，超出部分由 Windows
WDDM 用系统内存顶上，之后每次访问都走 PCIe，于是：

  · 1 个 epoch + 20 张验证 + 20 张测试 = 6.99 分钟（预期约 2 分钟）
  · 测试评测 20 张切片跑了 4 分钟仍未结束（预期 10 秒）

而单独的 `probe_eval_mem.py` 只测到 **491 MB**。两者矛盾，必须定位。

本脚本的关键区别：**不重置峰值**，累积报告 max_memory_allocated，
并按训练/评测的每个阶段拆开——复现真实训练路径（autocast + GradScaler +
clip_grad_norm_），而不是只测裸前向。

用法
------------------------------------------------------------------
    python experiments/probe_phase_mem.py
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


def snap(label, t0=None):
    torch.cuda.synchronize()
    a = mb(torch.cuda.memory_allocated())
    p = mb(torch.cuda.max_memory_allocated())
    r = mb(torch.cuda.memory_reserved())
    dt = f"  {time.time()-t0:6.2f}s" if t0 else "        "
    print(f"  {label:<38}{dt}  当前 {a:8.1f}  峰值 {p:8.1f}  预留 {r:8.1f} MB",
          flush=True)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    dev = torch.device("cuda")
    print(f"GPU {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB\n",
          flush=True)

    model = REDCNN().to(dev)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    snap("模型+优化器建好")

    def train_step(bs, patch):
        x = torch.rand(bs, 1, patch, patch, device=dev)
        y = torch.rand(bs, 1, patch, patch, device=dev)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return x, y

    model.train()
    t0 = time.time()
    train_step(8, 128)
    snap("训练 1 步  bs=8 patch=128", t0)

    t0 = time.time()
    for _ in range(19):
        train_step(8, 128)
    snap("训练 20 步（累计）", t0)

    t0 = time.time()
    for _ in range(180):
        train_step(8, 128)
    snap("训练 200 步（=1 个 epoch）", t0)

    # ---- 评测：复刻 evaluate() 的写法 ----
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        out = model(torch.rand(2, 1, 512, 512, device=dev))
        snap("评测 1 步  512^2 bs=2（裸前向）", t0)

        t0 = time.time()
        for _ in range(9):
            o = model(torch.rand(2, 1, 512, 512, device=dev))
            g = torch.rand(2, 1, 512, 512, device=dev)
            p = torch.clamp(o * 4096.0 - 1024.0, -160.0, 240.0)
            gg = torch.clamp(g * 4096.0 - 1024.0, -160.0, 240.0)
            _ = (p - gg).pow(2).flatten(1).mean(1)
            from utils.metrics import ssim_torch
            _ = ssim_torch(o, g)
        snap("评测 10 步  512^2 bs=2（完整）", t0)

    print("\n--- 分配器明细 ---", flush=True)
    print(torch.cuda.memory_summary(abbreviated=True), flush=True)

    print("\n--- 归一化：每一步实际驻留多少 ---", flush=True)
    print("    若「训练 200 步」的峰值远大于「训练 1 步」，说明**有东西在累积**；",
          flush=True)
    print("    若两者接近，则峰值来自单步，问题在单步的瞬时分配。", flush=True)


if __name__ == "__main__":
    main()
