"""
评测批大小（batch_eval）是否影响数值。

为什么必须验
------------------------------------------------------------------
2026-09-14 实测：RED-CNN 在 512×512 上，batch_eval 从 1 涨到 2，
耗时跳 **49 倍**、显存跳 **14 倍**（0.70 s / 489 MB → 34.17 s / 7025 MB），
根因是 cuDNN 为 ConvTranspose2d 挑了个 workspace 约 7 GB 的算法。
见 `experiments/probe_eval_batch.py`。

于是把 `train.evaluate()` 对参数量大的模型改成 batch_eval=1。
**但这只有在"批大小不改变数值"时才成立。** 若两者结果不同，
这个"优化"就是在悄悄改实验数据 —— 本项目已经因为
"把推理当事实"栽过跟头（见 ../MGF-Net_项目教训）。

本测试直接调用真的 `train.evaluate()`，同一模型、同一批切片，
只换 batch_eval，逐切片比对 PSNR / SSIM。

跑法
------------------------------------------------------------------
    python tests/test_eval_batch_equiv.py
    python -m pytest tests/test_eval_batch_equiv.py -s

（无 pytest 也能跑：文件末尾有 main。）
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import train as T                          # noqa: E402
from models.redcnn import REDCNN           # noqa: E402

N_SLICES = 4
SIZE = 512
TOL_PSNR = 1e-3      # dB。远小于论文里任何有意义的差异（最小效应量 0.27 dB）
TOL_SSIM = 1e-6


def _make_loader(n, size, seed):
    """造 n 张 [0,1] 的 (obs, gt) 对。固定种子，保证两次评测看到同样的图。"""
    g = torch.Generator().manual_seed(seed)
    obs = torch.rand(n, 1, size, size, generator=g)
    gt = torch.rand(n, 1, size, size, generator=g)
    return DataLoader(TensorDataset(obs, gt), batch_size=1, shuffle=False)


def _eval_with(model, loader, batch_eval, device):
    rows, mean = T.evaluate(model, loader, device, limit=None,
                            batch_eval=batch_eval)
    return rows, mean


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if not torch.cuda.is_available():
        print("没有 CUDA，跳过（本测试针对 GPU 评测路径）")
        return 0

    device = torch.device("cuda")
    torch.manual_seed(0)
    model = REDCNN().to(device).eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"RED-CNN {n_par:,} 参数，{N_SLICES} 张 {SIZE}×{SIZE}\n")

    loader = _make_loader(N_SLICES, SIZE, seed=1234)

    print("  batch_eval=1 评测中 …", flush=True)
    r1, m1 = _eval_with(model, loader, 1, device)
    print("  batch_eval=2 评测中 …（已知会慢约 50 倍）", flush=True)
    r2, m2 = _eval_with(model, loader, 2, device)

    print(f"\n  {'#':<4}{'PSNR(bs=1)':>14}{'PSNR(bs=2)':>14}{'Δ':>12}"
          f"{'SSIM(bs=1)':>14}{'SSIM(bs=2)':>14}{'Δ':>12}")
    print("  " + "-" * 82)
    d_psnr, d_ssim = [], []
    for i, (a, b) in enumerate(zip(r1, r2)):
        dp = abs(a["PSNR"] - b["PSNR"])
        ds = abs(a["SSIM"] - b["SSIM"])
        d_psnr.append(dp)
        d_ssim.append(ds)
        print(f"  {i:<4}{a['PSNR']:>14.6f}{b['PSNR']:>14.6f}{dp:>12.2e}"
              f"{a['SSIM']:>14.8f}{b['SSIM']:>14.8f}{ds:>12.2e}")

    mx_p, mx_s = max(d_psnr), max(d_ssim)
    print("  " + "-" * 82)
    print(f"  最大差：PSNR {mx_p:.3e} dB   SSIM {mx_s:.3e}")
    print(f"  均值差：PSNR {abs(m1['PSNR']-m2['PSNR']):.3e} dB   "
          f"SSIM {abs(m1['SSIM']-m2['SSIM']):.3e}")

    ok = mx_p < TOL_PSNR and mx_s < TOL_SSIM
    print(f"\n  {'✅ 通过' if ok else '❌ 失败'}："
          f"批大小{'不影响' if ok else '**影响**'}数值"
          f"（阈值 PSNR<{TOL_PSNR} dB, SSIM<{TOL_SSIM}）")
    return 0 if ok else 1


def test_batch_equiv():
    """pytest 入口。"""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
