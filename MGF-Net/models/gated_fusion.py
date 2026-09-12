"""
v2 Cross-Modal Gated Fusion — Residual-style fusion blocks.

Key change from v1:
- v1: fused = sigmoid(gate) * MRI + (1-gate) * CT   (convex combo, loses info)
- v2: fused = CT + tanh(CNN(CT,MRI)) * (MRI - CT)    (residual, preserves CT base)
  The CNN learns WHERE and HOW MUCH MRI detail to inject into the CT base.

Each FusionBlock has ~8.7K params with 32 internal channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


GATE_TYPES = ("residual_capped", "neutral_sigmoid")


class FusionBlock(nn.Module):
    """单个频段的跨模态融合块。

    支持两种门控形式，**用于受控对照实验**（见 `experiments/`）：

    ``residual_capped``（v2 原式，默认）
        fused = CT + tanh(g) * 0.4 * (MRI - CT)
        等价于 (1-w)*CT + w*MRI，其中 w ∈ [-0.4, 0.4]。
        **MRI 的系数被手写的 0.4 卡住**，等权平均 (w=0.5) 在数学上不可达。
        审计实测：checkpoint 训练后 w 的均值 0.3983、25 分位即达上限 0.4，
        即门控网络实际在请求更多 MRI 而被该常数摁住。

    ``neutral_sigmoid``（对照）
        fused = (1-w)*CT + w*MRI,  w = sigmoid(g)，w ∈ (0, 1)
        无人工上限，两端都能取到。注意最后一层激活随之改为恒等（输出 logits），
        否则 tanh 后再 sigmoid 会把 w 压进 [0.269, 0.731] 的窄带，反而更不中性。

    两者的 ``refine`` 分支保持完全一致，保证对照只差这一个变量。

    Args:
        in_ch: 每模态输入通道数（灰度图为 1）
        mid_ch: 内部特征通道数
        gate_type: 'residual_capped' | 'neutral_sigmoid'
    """

    def __init__(self, in_ch=1, mid_ch=32, gate_type="residual_capped"):
        super().__init__()
        if gate_type not in GATE_TYPES:
            raise ValueError(f"未知 gate_type={gate_type!r}，可选 {GATE_TYPES}")
        self.gate_type = gate_type

        # 权重预测网络。末层激活随门控类型变化：
        #   residual_capped -> Tanh，配合外部的 *0.4 得到 [-0.4, 0.4]
        #   neutral_sigmoid -> Identity，输出 logits，外部再 sigmoid
        final_act = nn.Tanh() if gate_type == "residual_capped" else nn.Identity()
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_ch * 2, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1),
            final_act,
        )

        # 融合后的精炼。两种门控共用同一结构，保证对照公平
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch // 2, 3, padding=1),
            nn.BatchNorm2d(mid_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 2, in_ch, 3, padding=1),
        )

    def gate_weights(self, ct, mri):
        """返回 MRI 侧的等效系数 w（诊断用）。

        residual_capped: w = tanh(g)*0.4
        neutral_sigmoid: w = sigmoid(g)
        """
        raw = self.weight_net(torch.cat([ct, mri], dim=1))
        return raw * 0.4 if self.gate_type == "residual_capped" else torch.sigmoid(raw)

    def forward(self, ct, mri):
        if self.gate_type == "residual_capped":
            w = self.gate_weights(ct, mri)          # [-0.4, 0.4]
        else:
            w = self.gate_weights(ct, mri)          # (0, 1)
        fused = ct + w * (mri - ct)                 # 等价于 (1-w)*CT + w*MRI
        fused = fused + self.refine(fused)
        return fused


class CrossBandAttention(nn.Module):
    """Lightweight SE-style attention across the 7 frequency subbands.

    Lets the model learn correlations between bands:
    e.g., if HH has strong edges, maybe reduce LH weight.
    """

    def __init__(self, n_bands=7, reduction=4):
        super().__init__()
        mid = max(n_bands // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(n_bands, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_bands),
            nn.Sigmoid()
        )

    def forward(self, band_features):
        """band_features: list of [B, C, H, W] tensors (7 subbands)."""
        # Global average pool each band → [B, C]
        stats = torch.stack([f.mean(dim=[2, 3]) for f in band_features], dim=1)  # [B, 7, C]
        stats = stats.mean(dim=2)  # [B, 7]  average over channels
        weights = self.fc(stats)    # [B, 7]
        # Apply per-band weights
        return [band_features[i] * weights[:, i:i+1, None, None] for i in range(len(band_features))]


class GatedFusionModule(nn.Module):
    """v2: 7-band residual fusion with cross-band attention.

    Architecture:
        For each of 7 subbands: FusionBlock(CT_sub, MRI_sub) → fused_sub
        CrossBandAttention across 7 fused subbands
        Return dict of fused subbands
    """

    def __init__(self, in_ch=1, mid_ch=32, gate_type="residual_capped"):
        super().__init__()
        self.gate_type = gate_type
        band_names = ['LL2', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1']
        self.band_names = band_names
        self.n_bands = len(band_names)

        # One FusionBlock per frequency band
        for name in band_names:
            setattr(self, f'fuse_{name}', FusionBlock(in_ch, mid_ch, gate_type))

        # Cross-band interaction
        self.cross_attn = CrossBandAttention(self.n_bands)

    def forward(self, ct_subbands, mri_subbands):
        """Input: dicts of subband tensors. Output: dict of fused subbands."""
        fused = {}

        # Independent fusion per band
        for name in self.band_names:
            fb = getattr(self, f'fuse_{name}')
            fused[name] = fb(ct_subbands[name], mri_subbands[name])

        # Cross-band attention
        band_list = [fused[name] for name in self.band_names]
        attended = self.cross_attn(band_list)

        return {name: attended[i] for i, name in enumerate(self.band_names)}
