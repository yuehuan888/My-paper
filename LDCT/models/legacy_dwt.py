"""
旧的分离式可学习小波（自 MGF-Net 迁移，仅作对照）。

⚠️ 本模块的正变换与逆变换是**两组互相独立的可学习参数**，无任何闭环保证。
MGF-Net 项目实测其训练后闭环相对误差达 60.9%（丢六成信号）。
在 LDCT 项目中它作为"无约束可学习"的对照臂，用于验证 PR-LWT 是否真有收益。

Multi-Level Learnable DWT / IDWT for frequency decomposition.

v2: Supports 2-level cascade for 7-subband decomposition.
    Level 1: input → LL1, LH1, HL1, HH1  (each H/2 × W/2)
    Level 2: LL1  → LL2, LH2, HL2, HH2  (each H/4 × W/4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _haar_filters():
    lo = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)], dtype=np.float32)
    hi = np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2)], dtype=np.float32)
    ll = np.outer(lo, lo); lh = np.outer(lo, hi)
    hl = np.outer(hi, lo); hh = np.outer(hi, hi)
    return torch.from_numpy(np.stack([ll, lh, hl, hh])[:, None, :, :])  # [4,1,2,2]


class DWTLayer(nn.Module):
    """Single-level DWT. decompose(input) → (LL, LH, HL, HH)."""
    def __init__(self, learnable=True):
        super().__init__()
        if learnable:
            self.filters = nn.Parameter(_haar_filters())
        else:
            self.register_buffer('filters', _haar_filters())

    def forward(self, x):
        B, C, H, W = x.shape
        xr = x.reshape(B*C, 1, H, W)
        outs = []
        for i in range(4):
            f = self.filters[i:i+1].repeat(C, 1, 1, 1)
            outs.append(F.conv2d(xr, f, stride=2, groups=C))
        return tuple(o.reshape(B, C, H//2, W//2) for o in outs)


class IDWTLayer(nn.Module):
    """Single-level inverse DWT. reconstruct(LL, LH, HL, HH) → image."""
    def __init__(self, learnable=True):
        super().__init__()
        if learnable:
            self.filters = nn.Parameter(_haar_filters())
        else:
            self.register_buffer('filters', _haar_filters())

    def forward(self, ll, lh, hl, hh):
        B, C, H, W = ll.shape
        out = 0
        for i, sub in enumerate([ll, lh, hl, hh]):
            sr = sub.reshape(B*C, 1, H, W)
            f = self.filters[i:i+1].repeat(C, 1, 1, 1)
            out = out + F.conv_transpose2d(sr, f, stride=2, groups=C)
        return out.reshape(B, C, 2*H, 2*W)


class MultiLevelDWT(nn.Module):
    """2-Level DWT: returns 7 subbands.
    Returns dict: {'LL2','LH2','HL2','HH2','LH1','HL1','HH1'}
    """
    def __init__(self, learnable=True):
        super().__init__()
        self.dwt1 = DWTLayer(learnable)
        self.dwt2 = DWTLayer(learnable)

    def forward(self, x):
        ll1, lh1, hl1, hh1 = self.dwt1(x)          # H/2
        ll2, lh2, hl2, hh2 = self.dwt2(ll1)         # H/4
        return {'LL2': ll2, 'LH2': lh2, 'HL2': hl2, 'HH2': hh2,
                'LH1': lh1, 'HL1': hl1, 'HH1': hh1}


class MultiLevelIDWT(nn.Module):
    """2-Level inverse DWT: 7 subbands → image."""
    def __init__(self, learnable=True):
        super().__init__()
        self.idwt1 = IDWTLayer(learnable)
        self.idwt2 = IDWTLayer(learnable)

    def forward(self, subbands):
        # Level 2: LL2+LH2+HL2+HH2 → LL1 recon
        ll1_recon = self.idwt2(subbands['LL2'], subbands['LH2'],
                               subbands['HL2'], subbands['HH2'])
        # Level 1: LL1_recon+LH1+HL1+HH1 → full image
        return self.idwt1(ll1_recon, subbands['LH1'],
                          subbands['HL1'], subbands['HH1'])
