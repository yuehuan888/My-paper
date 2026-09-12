"""
MGF-Net v2 — Multi-Level Frequency Fusion with Residual Injection.

Key improvements over v1:
1. 2-level DWT → 7 frequency subbands (LL2/LH2/HL2/HH2 + LH1/HL1/HH1)
2. Residual fusion: CT + tanh(CNN(CT,MRI)) * (MRI - CT) instead of convex combo
3. 32-channel FusionBlocks (~8.7K params each × 7 ≈ 61K total)
4. Cross-band SE attention for inter-frequency coordination
5. Edge-aware refinement (from v1)

Target: ~65K parameters, <0.5s inference on RTX 3050.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dwt_layer import MultiLevelDWT, MultiLevelIDWT
from .gated_fusion import GatedFusionModule
from .edge_refine import EdgeRefineModule


class MGFNet(nn.Module):
    """MGF-Net v2: Multi-scale Gated Frequency Fusion Network.

    Args:
        in_channels: 1 for grayscale medical images
        mid_channels: channels inside FusionBlocks (default 32)
        learnable_dwt: train wavelet filters end-to-end
        gate_type: 'residual_capped'（v2 原式）| 'neutral_sigmoid'（对照）
                   详见 models/gated_fusion.py
    """

    def __init__(self, in_channels=1, mid_channels=32, learnable_dwt=True,
                 gate_type="residual_capped"):
        super().__init__()

        # 2-level frequency decomposition → 7 subbands
        self.dwt = MultiLevelDWT(learnable=learnable_dwt)
        self.idwt = MultiLevelIDWT(learnable=learnable_dwt)

        # Residual fusion per band + cross-band attention
        self.fusion = GatedFusionModule(in_ch=in_channels, mid_ch=mid_channels,
                                        gate_type=gate_type)

        # Edge-aware refinement (from v1)
        self.edge_refine = EdgeRefineModule(in_channels=in_channels)

    def forward(self, ct, mri):
        """
        Args:
            ct:  [B, 1, H, W] CT image  (normalized [0,1])
            mri: [B, 1, H, W] MRI image (normalized [0,1])
        Returns:
            fused: [B, 1, H, W]
        """
        H_in, W_in = ct.shape[2], ct.shape[3]

        # Pad to multiple of 4 (for 2-level DWT)
        pad_h = (4 - H_in % 4) % 4
        pad_w = (4 - W_in % 4) % 4
        if pad_h > 0 or pad_w > 0:
            ct = F.pad(ct, (0, pad_w, 0, pad_h), mode='reflect')
            mri = F.pad(mri, (0, pad_w, 0, pad_h), mode='reflect')

        # Step 1: Multi-level frequency decomposition
        ct_bands = self.dwt(ct)
        mri_bands = self.dwt(mri)

        # Step 2: Residual fusion per band + cross-band attention
        fused_bands = self.fusion(ct_bands, mri_bands)

        # Step 3: Multi-level IDWT reconstruction
        fused_init = self.idwt(fused_bands)

        # Crop back to original size
        if pad_h > 0 or pad_w > 0:
            fused_init = fused_init[:, :, :H_in, :W_in]
            ct = ct[:, :, :H_in, :W_in]
            mri = mri[:, :, :H_in, :W_in]

        # Step 4: Edge-aware refinement
        fused = self.edge_refine(fused_init, ct, mri)

        return fused


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
