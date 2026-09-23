"""
PR-Wavelet 去噪网络。

核心思路
------------------------------------------------------------------
在 **可逆的小波域** 做去噪：

    含噪 CT x
      ↓  PR-LWT 分解（2 级，7 子带）—— 正逆变换**共享同一组 P/U**
      ↓  逐子带残差预测：delta_b = head_b(band_b)
      ↓  清理后的子带：clean_b = band_b − delta_b
      ↓  PR-LWT 重建
    干净 CT x̂

为什么是"可逆"小波而非普通小波
------------------------------------------------------------------
小波域去噪的经典做法，但**变换本身必须是可逆的**才有意义：
若变换在正逆之间丢信息，网络就得一边去噪一边补偿变换损失，
而后者是"不该由网络承担的负担"。

PR-LWT 的可逆性由**结构**保证（逆变换不是另一组参数，而是把同一组 P/U
反向减回去），且提升滤波器的 taps 做了有界化以保证 float32 下的数值稳定。
实测闭环相对误差 1.40e-07（对比旧式分离参数实现的 60.9%）。

**这不自动意味着去噪效果更好**——上一个项目的教训是
"机制上更干净" ≠ "指标上更好"。故本模型提供 `use_pr_wavelet=False`
的对照开关，用于在同一实验中验证该设计是否真有收益。

参数效率
------------------------------------------------------------------
变换部分仅 24 个参数；主体是 7 个轻量预测头。默认配置总参数量约 10K 量级，
契合"轻量"的叙事，也便于在 4 GB 显存上训练。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lifting_dwt import MultiLevelLifting


class BandHead(nn.Module):
    """单个子带的残差预测头。

    输入输出均为单通道，中间用 `mid_ch` 个通道做两层级联。
    末尾无激活——预测的是残差（可正可负）。
    """

    def __init__(self, mid_ch: int = 16, n_conv: int = 2):
        super().__init__()
        layers = []
        c_in = 1
        for i in range(n_conv):
            c_out = mid_ch if i < n_conv - 1 else 1
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=True)]
            if i < n_conv - 1:
                layers += [nn.ReLU(inplace=True)]
            c_in = c_out
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PRWaveletDenoiser(nn.Module):
    """基于 PR-LWT 的小波域去噪网络。

    Args:
        levels:    小波分解级数（2 级 → 7 子带）
        mid_ch:    每个预测头的内部通道数
        n_conv:    每个预测头的卷积层数
        wavelet:
            'pr'            → PR-LWT（提升格式，正逆共享 P/U，可逆性由结构保证）
            'fixed'         → 固定正交归一 Haar（不可学习）
            'unconstrained' → **旧式分离参数**的可学习小波（无闭环保证）

            ⚠️ 三臂对照的意义：
              pr vs unconstrained —— 是**可逆性是否带来收益**（本项目要回答的问题）
              pr vs fixed         —— 是**可学习是否带来收益**
            两者是不同的问题，不能混为一谈。

            ⚠️ 早先本类的 `use_pr_wavelet=False` 只把 lifting 设为不可学习，
            **并未切换到旧式实现**，导致"对照臂"实际测的是固定 Haar。
            该误标已修正为显式三选一。
        global_residual:
            是否在图像域再叠一层全局残差分支
            （小波域只处理各子带，图像域残差可补偿整体强度偏移）
    """

    # ⚠️ 2026-09-16：由硬编码 7 个（=levels 2）改为**按 levels 动态**，与
    #    lifting_dwt.MultiLevelLifting.band_names 同一来源，避免两处各自硬编码而漂移。
    @staticmethod
    def band_names(levels: int) -> tuple:
        from models.lifting_dwt import MultiLevelLifting
        return MultiLevelLifting.band_names(levels)

    def __init__(self, levels: int = 2, mid_ch: int = 16, n_conv: int = 2,
                 wavelet: str = "pr", global_residual: bool = False,
                 synth_mismatch: float = 0.0, n_taps: int = 3, bound: float = 0.5,
                 init_drift: float = 0.0):
        super().__init__()
        if wavelet not in ("pr", "fixed", "unconstrained"):
            raise ValueError(f"未知 wavelet={wavelet!r}")
        self.levels = levels
        self.wavelet_kind = wavelet
        self.global_residual = global_residual
        # ---- 提升格式的两个旋钮（**贡献 3 的消融用**）----------------------
        # n_taps: 每个 P/U 滤波器的抽头数（默认 3，与全部已有结果一致）
        # bound : taps 相对 Haar 初始值的最大偏离（|taps| ≤ init + bound）
        #
        # ⚠️ 二者默认值 = 现有行为，改动对已有结果**零影响**。
        #    bound 是论文贡献 3 的核心：无界化会让 float32 闭环从 3.6e-07
        #    退化到 5.2e+00。此前该结论只有**单个观测点**，无法画曲线；
        #    暴露此参数后才能做 bound 扫描，把"需要界定"从断言变成剂量-响应。
        self.n_taps = int(n_taps)
        self.bound = float(bound)
        # E3 用：把 taps 初始化在离 Haar 距离 init_drift 处。
        # 默认 0 = 严格 Haar 初始化，**对已有结果零影响**。
        self.init_drift = float(init_drift)
        if n_taps % 2 == 0:
            raise ValueError(f"n_taps 必须为奇数（零填充需要中心抽头），得到 {n_taps}")
        if bound < 0:
            raise ValueError(f"bound 必须非负，得到 {bound}")
        # 受控闭环误差注入：>0 时，PR 臂的**合成**用 (1−ε)·U 而非 U。
        # 用于干预实验——检验"闭环误差本身是否导致性能退化"。
        # 默认 0.0，对已有结果零影响。
        self.synth_mismatch = float(synth_mismatch)
        if synth_mismatch and wavelet != "pr":
            raise ValueError("synth_mismatch 只对 wavelet='pr' 有意义")

        if wavelet == "unconstrained":
            from .legacy_dwt import MultiLevelDWT, MultiLevelIDWT
            self.dwt = MultiLevelDWT(learnable=True)
            self.idwt = MultiLevelIDWT(learnable=True)
            self.wavelet = None
        else:
            self.wavelet = MultiLevelLifting(levels=levels, n_taps=self.n_taps,
                                             learnable=(wavelet == "pr"),
                                             bound=self.bound,
                                             synth_mismatch=synth_mismatch,
                                             init_drift=self.init_drift)
            self.dwt = self.idwt = None
        # 子带名随 levels 变化（levels=2 -> 7 个子带，levels=3 -> 10 个）。
        # 头数与子带数保持一致，故这里必须用实例级的名字列表。
        self.bands = self.band_names(levels)
        self.heads = nn.ModuleDict({b: BandHead(mid_ch, n_conv) for b in self.bands})

        if global_residual:
            self.global_branch = nn.Sequential(
                nn.Conv2d(1, mid_ch, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, 1, 3, padding=1),
            )
        else:
            self.global_branch = None

    # ------------------------------------------------------------ 频域变换
    def decompose(self, x):
        if self.wavelet_kind == "unconstrained":
            return self.dwt(x)
        return self.wavelet.decompose(x)

    def reconstruct(self, bands):
        if self.wavelet_kind == "unconstrained":
            return self.idwt(bands)
        return self.wavelet.reconstruct(bands)

    def _pad_to_multiple(self, x):
        """按 2^levels 对齐边长。

        ⚠️ LoDoPaB 的 362 是偶数，但 362/2 = 181 是**奇数**，第二级分解会因
        偶奇切分长度不等而报错。故必须按 2^levels（而非 2）对齐。
        返回 (padded, (H, W))。
        """
        H, W = x.shape[-2], x.shape[-1]
        m = 2 ** self.levels
        ph, pw = (m - H % m) % m, (m - W % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        return x, (H, W)

    def transform_roundtrip(self, x):
        """闭环诊断：返回 (max|err|, rel_L2)。内部自行对齐尺寸。"""
        xp, (H, W) = self._pad_to_multiple(x)
        with torch.no_grad():
            rec = self.reconstruct(self.decompose(xp))[..., :H, :W]
            ref = x[..., :H, :W]
            d = (rec - ref).abs()
            rel = (torch.norm(d) / torch.norm(ref)).item() if torch.norm(ref) > 0 else 0.0
            return d.max().item(), rel

    # ------------------------------------------------------------ 前向
    def forward(self, x):
        """x: [B, 1, H, W]，值域 [0,1]。返回同形状的干净估计。"""
        xp, (H, W) = self._pad_to_multiple(x)

        bands = self.decompose(xp)
        cleaned = {b: bands[b] - self.heads[b](bands[b]) for b in self.bands}
        out = self.reconstruct(cleaned)

        if self.global_branch is not None:
            out = out + self.global_branch(xp)

        return out[..., :H, :W]

    # ------------------------------------------------------------ 诊断
    def param_report(self) -> dict:
        # unconstrained 臂的正/逆变换是**两个独立模块**，都要计入
        if self.wavelet_kind == "unconstrained":
            n_wave = (sum(p.numel() for p in self.dwt.parameters())
                      + sum(p.numel() for p in self.idwt.parameters()))
        else:
            n_wave = sum(p.numel() for p in self.wavelet.parameters())
        n_head = sum(p.numel() for p in self.heads.parameters())
        n_glob = (sum(p.numel() for p in self.global_branch.parameters())
                  if self.global_branch is not None else 0)
        total = sum(p.numel() for p in self.parameters())
        return {"wavelet": n_wave, "heads": n_head, "global": n_glob, "total": total,
                "wavelet_frac": n_wave / max(total, 1)}


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = PRWaveletDenoiser()
    r = m.param_report()
    print("PRWaveletDenoiser 参数构成：")
    for k, v in r.items():
        print(f"  {k:<14} {v:,.0f}" if isinstance(v, float) is False else f"  {k:<14} {v:.4f}")

    x = torch.rand(2, 1, 362, 362)
    y = m(x)
    print(f"\n前向：{tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"输出值域: [{y.min():.4f}, {y.max():.4f}]")
    me, rel = m.transform_roundtrip(x)
    print(f"变换闭环: max|err|={me:.3e}  rel_L2={rel:.3e}")
