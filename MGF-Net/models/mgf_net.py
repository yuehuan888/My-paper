"""
MGF-Net — 多尺度门控频域融合网络。

v2 架构：2 级频域分解 → 7 子带 → 逐子带门控融合 → 跨频段协调 → 逆变换 → 边缘精炼。

两个可切换的维度（用于受控对照实验，见 `experiments/`）：

`wavelet` —— 频域变换的实现方式
    'legacy'  旧实现：正/逆变换是**两组独立**可学习参数（`dwt_layer.py`），
              无任何闭环保证。实测 checkpoint 的闭环 rel_L2 达 60.9%。
    'lifting' PR-LWT：提升格式，正逆**共用同一组 P/U**，可逆性由结构保证，
              且 taps 有界化以保证 float32 下的数值稳定（`lifting_dwt.py`）。

`gate_type` —— 跨模态融合的门控形式
    'residual_capped'  旧式：fused = CT + tanh(g)·0.4·(MRI−CT)，w ∈ [−0.4, 0.4]。
                       实测 99.13% 的像素贴在 +0.4 上限，门控饱和为常数，
                       融合退化成固定 60/40 线性混合。
    'neutral_sigmoid'  中性：fused = (1−w)·CT + w·MRI，w = sigmoid(g) ∈ (0,1)。
                       实测权重铺满 [0.007, 0.977]，标准差 9.4 倍于旧式。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dwt_layer import MultiLevelDWT, MultiLevelIDWT
from .gated_fusion import GatedFusionModule
from .edge_refine import EdgeRefineModule
from .lifting_dwt import MultiLevelLifting

WAVELET_KINDS = ("legacy", "lifting")


class MGFNet(nn.Module):
    """MGF-Net：多尺度门控频域融合网络。

    Args:
        in_channels: 输入通道数，灰度为 1
        mid_channels: FusionBlock 内部通道数
        learnable_dwt: 频域滤波器是否端到端可学习
        gate_type: 'residual_capped' | 'neutral_sigmoid'，见模块文档
        wavelet: 'legacy' | 'lifting'，见模块文档
    """

    def __init__(self, in_channels=1, mid_channels=32, learnable_dwt=True,
                 gate_type="residual_capped", wavelet="legacy"):
        super().__init__()
        if wavelet not in WAVELET_KINDS:
            raise ValueError(f"未知 wavelet={wavelet!r}，可选 {WAVELET_KINDS}")
        self.wavelet_kind = wavelet
        self.gate_type = gate_type

        if wavelet == "lifting":
            # 正变换与逆变换是**同一个模块的方法**，共享 P/U —— PR 的结构来源
            self.wavelet = MultiLevelLifting(levels=2, n_taps=3,
                                             learnable=learnable_dwt)
            self.dwt = None
            self.idwt = None
        else:
            # 旧实现：两组互相独立的参数，无闭环保证（保留用于对照）
            self.dwt = MultiLevelDWT(learnable=learnable_dwt)
            self.idwt = MultiLevelIDWT(learnable=learnable_dwt)
            self.wavelet = None

        self.fusion = GatedFusionModule(in_ch=in_channels, mid_ch=mid_channels,
                                        gate_type=gate_type)
        self.edge_refine = EdgeRefineModule(in_channels=in_channels)

    # ------------------------------------------------------------ 频域变换
    def decompose(self, x: torch.Tensor) -> dict:
        """统一入口：返回 7 个子带的 dict。"""
        if self.wavelet_kind == "lifting":
            return self.wavelet.decompose(x)
        return self.dwt(x)

    def reconstruct(self, bands: dict) -> torch.Tensor:
        """统一入口：由 7 个子带重建图像。"""
        if self.wavelet_kind == "lifting":
            return self.wavelet.reconstruct(bands)
        return self.idwt(bands)

    def transform_roundtrip(self, x: torch.Tensor):
        """闭环诊断：decompose → reconstruct，返回 (max|err|, rel_L2)。"""
        with torch.no_grad():
            rec = self.reconstruct(self.decompose(x))
            d = (rec - x).abs()
            rel = (torch.norm(d) / torch.norm(x)).item() if torch.norm(x) > 0 else 0.0
            return d.max().item(), rel

    # ------------------------------------------------------------ 前向
    def forward(self, ct, mri):
        """
        Args:
            ct:  [B, 1, H, W] CT 图，值域 [0,1]
            mri: [B, 1, H, W] MRI 图，值域 [0,1]
        Returns:
            fused: [B, 1, H, W]
        """
        H_in, W_in = ct.shape[2], ct.shape[3]

        # 两级变换要求边长是 4 的倍数
        pad_h = (4 - H_in % 4) % 4
        pad_w = (4 - W_in % 4) % 4
        if pad_h > 0 or pad_w > 0:
            ct = F.pad(ct, (0, pad_w, 0, pad_h), mode='reflect')
            mri = F.pad(mri, (0, pad_w, 0, pad_h), mode='reflect')

        ct_bands = self.decompose(ct)
        mri_bands = self.decompose(mri)
        fused_bands = self.fusion(ct_bands, mri_bands)
        fused_init = self.reconstruct(fused_bands)

        if pad_h > 0 or pad_w > 0:
            fused_init = fused_init[:, :, :H_in, :W_in]
            ct = ct[:, :, :H_in, :W_in]
            mri = mri[:, :, :H_in, :W_in]

        return self.edge_refine(fused_init, ct, mri)


def count_parameters(model):
    """可训练参数量。

    ⚠️ 论文中应同时报告**总参数量**（含 buffer）与可训练参数量，
    以及推理显存与 MACs/FLOPs，口径需写明。这里只给可训练量。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
