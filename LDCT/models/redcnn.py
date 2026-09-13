"""
RED-CNN 基线（Chen et al., "Low-dose CT via convolutional neural network",
Biomedical Optics Express, 2017）。

为什么需要它
------------------------------------------------------------------
本工作最大的短板是"没有同划分的真实基线对比"——论文里的 RED-CNN/CTformer
数字均引自文献，而文献的训练划分（9 患者）与本工作（8 训练 + 1 验证）不同。
审稿人最可能据此拒稿。故在同一划分、同一评测口径、同一训练协议下
自行复现 RED-CNN，形成可直接对比的表格。

结构（按原论文）
------------------------------------------------------------------
  5 层 5×5 卷积（96 通道，ReLU） + 5 层对称 5×5 反卷积
  **无池化、无 BatchNorm** —— 保持空间分辨率，输入输出同尺寸
  padding = 2（5×5 核的一半向下取整）

参数量约 1.85 M，是本工作（2,159）的约 **860 倍**。

⚠️ 复现声明
------------------------------------------------------------------
本实现按原论文描述编写，但**未逐位核对作者原始代码**。
论文中引用其数字时应注明"我们自行复现"，并说明与原文报告的差异（若有）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class REDCNN(nn.Module):
    """RED-CNN：10 层全卷积编解码（无池化）。

    Args:
        in_channels:  输入通道数（灰度 CT 为 1）
        out_channels: 输出通道数
        n_filters:    内部通道数（原论文为 96）
        kernel:       卷积核尺寸（原论文为 5）
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 n_filters: int = 96, kernel: int = 5):
        super().__init__()
        pad = kernel // 2
        f = n_filters

        def conv(i, o):
            return nn.Conv2d(i, o, kernel, padding=pad, bias=True)

        def deconv(i, o):
            return nn.ConvTranspose2d(i, o, kernel, padding=pad,
                                      output_padding=0, bias=True)

        self.encoder = nn.Sequential(
            conv(in_channels, f), nn.ReLU(inplace=True),
            conv(f, f), nn.ReLU(inplace=True),
            conv(f, f), nn.ReLU(inplace=True),
            conv(f, f), nn.ReLU(inplace=True),
            conv(f, f), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            deconv(f, f), nn.ReLU(inplace=True),
            deconv(f, f), nn.ReLU(inplace=True),
            deconv(f, f), nn.ReLU(inplace=True),
            deconv(f, f), nn.ReLU(inplace=True),
            deconv(f, out_channels),          # 末层无激活
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = REDCNN()
    n = count_parameters(m)
    print(f"RED-CNN 参数量: {n:,}")
    print(f"  对比 PR-Wavelet: 2,159  →  倍数 {n/2159:.0f}x")

    # 尺寸与显存检查（4GB 卡能不能跑）
    for patch, bs in [(128, 8), (128, 4), (64, 8)]:
        try:
            x = torch.rand(bs, 1, patch, patch, device="cuda")
            mm = m.cuda()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                y = mm(x)
            peak = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  patch={patch} batch={bs}: 输出 {tuple(y.shape)}  "
                  f"前向峰值显存 {peak:.0f} MB")
            del x, y; torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"  patch={patch} batch={bs}: **显存不足**")
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  patch={patch} batch={bs}: {type(e).__name__}: {e}")
