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


class FusionBlock(nn.Module):
    """Residual-style fusion for a single frequency subband.

    Instead of choosing between CT and MRI, we treat CT as the base
    and learn to add MRI-specific details where they exist.
    Fused = CT + tanh(weight) * (MRI - CT) + refinement

    Args:
        in_ch: input channels per modality (1 for grayscale)
        mid_ch: internal feature channels (default 32)
    """

    def __init__(self, in_ch=1, mid_ch=32):
        super().__init__()

        # Weight prediction: where to inject MRI detail
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_ch * 2, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1),
            nn.Tanh()   # weight in [-1, 1]: -1=keep CT, +1=add MRI
        )

        # Refinement after fusion
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch // 2, 3, padding=1),
            nn.BatchNorm2d(mid_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch // 2, in_ch, 3, padding=1),
        )

    def forward(self, ct, mri):
        # Residual fusion: CT as base, capped MRI detail injection
        # weight in [-1,1] scaled to [-0.4, 0.4] → max 40% injection
        concat = torch.cat([ct, mri], dim=1)            # [B, 2C, H, W]
        weight = self.weight_net(concat) * 0.4           # capped residual
        diff = mri - ct                                   # MRI minus CT
        fused = ct + weight * diff                        # bounded injection
        fused = fused + self.refine(fused)                # final polish
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

    def __init__(self, in_ch=1, mid_ch=32):
        super().__init__()
        band_names = ['LL2', 'LH2', 'HL2', 'HH2', 'LH1', 'HL1', 'HH1']
        self.band_names = band_names
        self.n_bands = len(band_names)

        # One FusionBlock per frequency band
        for name in band_names:
            setattr(self, f'fuse_{name}', FusionBlock(in_ch, mid_ch))

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
