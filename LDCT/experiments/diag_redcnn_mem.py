"""
诊断 RED-CNN 在 4GB 卡上的显存与耗时分布。

背景：RED-CNN (1.85M 参数) 训练时显存会涨到 3881/4096 MiB，速度掉到
约 4.75 分钟/轮（预期 1.4 分钟）。本脚本分阶段测出显存花在哪。
"""

from __future__ import annotations

import os
import sys
import time

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from data.dataset import AapmDataset          # noqa: E402
from models.redcnn import REDCNN              # noqa: E402
import train as T                             # noqa: E402


def mem_mb():
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def main():
    dev = torch.device("cuda")
    tr = AapmDataset(ROOT + "/data/aapm_h5", ROOT + "/splits/aapm_mayo_3mm.json",
                     "train", patch_size=128, is_training=True, cache=True)
    va = AapmDataset(ROOT + "/data/aapm_h5", ROOT + "/splits/aapm_mayo_3mm.json",
                     "val", cache=True)
    m = REDCNN().to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    crit = torch.nn.L1Loss()
    n_par = sum(p.numel() for p in m.parameters())
    print(f"RED-CNN 参数量 = {n_par:,}")

    # ---------- 1. 训练步 ----------
    dl = DataLoader(tr, batch_size=2, shuffle=True, num_workers=0,
                    pin_memory=True, drop_last=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    it = iter(dl)
    n, t0 = 0, time.time()
    while n < 50:
        try:
            o, g = next(it)
        except StopIteration:
            it = iter(dl)
            o, g = next(it)
        o, g = o.to(dev), g.to(dev)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            loss = crit(m(o), g)
        loss.backward()
        opt.step()
        n += 1
    torch.cuda.synchronize()
    t_train = (time.time() - t0) / 50 * 1000
    print(f"[训练] batch=2 patch=128: {t_train:.1f} ms/步   峰值显存 {mem_mb():.0f} MB")

    # ---------- 2. 验证（关键嫌疑）----------
    for be in [2, 1]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        vl = DataLoader(va, batch_size=1, shuffle=False)
        t0 = time.time()
        T.evaluate(m, vl, dev, limit=40, batch_eval=be)
        torch.cuda.synchronize()
        dt = time.time() - t0
        print(f"[验证] 40 张 512x512, batch_eval={be}: {dt:.1f}s   峰值显存 {mem_mb():.0f} MB")

    # ---------- 3. 整轮估算 ----------
    print()
    print(f"  单轮训练 400 步 × {t_train/1000:.3f}s = {t_train*400/60000:.1f} 分钟")
    print(f"  30 轮训练 = {t_train*400*30/60000:.1f} 分钟")
    print(f"  （若实测约 4.75 分钟/轮，说明另有开销，见上面的验证耗时）")


if __name__ == "__main__":
    main()
