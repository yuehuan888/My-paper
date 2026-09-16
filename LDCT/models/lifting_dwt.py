"""
PR-LWT：完美重构约束的可学习小波（Perfect-Reconstruction Learnable Wavelet）。

为什么需要它
================================================================
现有 `dwt_layer.py` 用两组**互相独立**的可学习滤波器做正变换与逆变换
（`MultiLevelDWT.dwt1/dwt2` 与 `MultiLevelIDWT.idwt1/idwt2`）。
Haar 能逆变换依赖分析滤波器正交，一旦两组参数各自漂移，正交性丢失，
闭环误差急剧上升。

实测（`diagnose_step1.py` Q2，checkpoint epoch=60）：

    随机图    max|err| = 0.6697   rel_L2 = 60.9%
    常量图    max|err| = 0.3695（DC 分量都不保持）
    Haar 对照 max|err| = 4.17e-07

即分解再重构走一圈，**六成信号丢失**。

本模块的做法
================================================================
用 **lifting scheme（提升格式）** 参数化：

    分解： x → (x_e, x_o)              # 仅偶奇切片，无卷积
           d = x_o − P(x_e)            # 预测步 → 细节
           s = x_e + U(d)              # 更新步 → 近似

    合成： x_e = s − U(d)              # P、U 与分解时**是同一组参数**
           x_o = d + P(x_e)
           x = interleave(x_e, x_o)

**可逆性由结构保证**：逆变换不是另一组独立参数，而是把同一个 P、U 反向减回去。
无论 P、U 被训练成什么值，`inverse(forward(x)) == x` 恒成立（浮点精度内）。

Haar 初始化
================================================================
取 P = 恒等（`[0,1,0]`）、U = 0.5·恒等（`[0,0.5,0]`）时：
    d = x_o − x_e,  s = (x_e + x_o) / 2
再施加 √2 缩放（`s ← √2·s`，`d ← d/√2`）即得**正交归一 Haar 系数**：
    LL = √2·s = (x_e + x_o)/√2,  H = d/√2 = (x_o − x_e)/√2
故 `normalize=True` 时初始化严格等价于固定 Haar。

⚠️ 注意：训练开始后 P、U 漂移，正交性不再保持（也不应保持——
自适应是目的）。但 **PR 依然成立**，这正是本设计的意义：
把可逆性从"希望它成立"变成"结构上必然成立"。

PR 的边界
================================================================
PR 只保证**系数未被修改**时的分析–合成闭环。融合会改变系数，
故 PR **不保证**融合信息全保留，更不保证诊断价值。不得称"无损融合"。

参数数量
================================================================
每个方向 2 个滤波器（P、U）× 3 taps = 6；两级 × 两方向 × 6 = **24 个**
（跨模态共享、无 bias、级与方向之间不共享时）。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _lift_taps(n_taps: int = 3, kind: str = "predict") -> torch.Tensor:
    """返回提升滤波器的初始 taps。

    kind='predict' → 恒等预测 [0,1,0]（P）
    kind='update'  → 半恒等     [0,0.5,0]（U），使 s = (x_e + x_o)/2
    """
    t = torch.zeros(n_taps, dtype=torch.float32)
    mid = n_taps // 2
    t[mid] = 1.0 if kind == "predict" else 0.5
    return t


class _LiftingBank1D(nn.Module):
    """一维提升的一对滤波器 P、U。正逆共用同一组参数。

    参数有界化（重要）
    ------------------------------------------------------------------
    实测（`tests/test_lifting_pr.py` 与专项诊断）：PR 在 float64 下无条件成立，
    但在 float32 下**条件数随参数幅度急剧恶化**——

        |P|,|U| |max| ≈ 1.0  → 闭环 rel_L2 ≈ 1.1e-07
        |P|,|U| |max| ≈ 2.0  → 闭环 rel_L2 ≈ 9.9e-07
        |P|,|U| |max| ≈ 3.5  → 闭环 rel_L2 ≈ 1.4e-03
        |P|,|U| |max| ≈ 5.3  → 闭环 rel_L2 ≈ 2.8e-01（灾难级）

    ⚠️ 2026-09-15 更正：本表早先的中间一行写「≈3.5 → max|err| ≈ 1e-04」——
       **与实测差约两个数量级**（实测 max|err| = 2.269e-02）。上表改为直接引用
       `tests/test_lifting_pr.py` 的【6】节实测输出（rel_L2 口径），可复算。

    同一组参数换 float64 复测，σ=3.0 时误差从 12.19 降到 6.3e-07，
    故这不是结构错误，而是浮点条件数问题。

    结论：**"有 PR 保证"不等于"数值稳定"**。不约束参数的可学习提升格式会漂到
    数值不可用的区间，退回与原实现相同的失败模式。

    故本实现把 taps 参数化为 `init + bound · tanh(θ)`，θ 可自由学习，
    但 taps 恒被限制在 `init ± bound` 内。默认 bound=0.5（即 |taps| ≤ 1.5），
    落在上述实测的安全区间；这样既保留自适应能力，又不破坏浮点闭环。
    """

    def __init__(self, n_taps: int = 3, learnable: bool = True,
                 bound: float = 0.5, synth_mismatch: float = 0.0):
        super().__init__()
        self.n_taps = n_taps
        self.bound = float(bound)
        # ---- 受控闭环误差注入（干预实验用）--------------------------------
        # synth_mismatch = ε > 0 时，**合成**用的 U 变成 (1−ε)·U，而分解仍用 U。
        # 这精确模拟了无约束臂的失效模式："逆变换不是正变换真正的逆"。
        # 误差随 ε 线性增长，故可标定到指定量级。
        #
        # ⚠️ 这不是可学习参数，是**固定注入**。默认 0.0 = 正常的 PR-LWT，
        #    行为与改动前完全一致（不影响任何已有结果）。
        self.synth_mismatch = float(synth_mismatch)
        p0 = _lift_taps(n_taps, "predict")
        u0 = _lift_taps(n_taps, "update")
        self.register_buffer("p_init", p0)
        self.register_buffer("u_init", u0)

        if learnable:
            # θ=0 时 tanh(0)=0，故初始严格等于 Haar 等价形式
            self.theta_P = nn.Parameter(torch.zeros(n_taps))
            self.theta_U = nn.Parameter(torch.zeros(n_taps))
        else:
            self.register_buffer("theta_P", torch.zeros(n_taps))
            self.register_buffer("theta_U", torch.zeros(n_taps))

    @property
    def P(self) -> torch.Tensor:
        return self.p_init + self.bound * torch.tanh(self.theta_P)

    @property
    def U(self) -> torch.Tensor:
        return self.u_init + self.bound * torch.tanh(self.theta_U)

    @property
    def U_synth(self) -> torch.Tensor:
        """**合成步**使用的 U。

        synth_mismatch=0 时就是 U 本身（严格 PR）；>0 时按 (1−ε) 缩放，
        从而使 reconstruct(decompose(x)) ≠ x，注入受控闭环误差。
        """
        if self.synth_mismatch == 0.0:
            return self.U
        return self.U * (1.0 - self.synth_mismatch)

    def drift(self) -> float:
        """参数相对 Haar 初始值的最大偏离量（诊断用）。"""
        with torch.no_grad():
            return float(max((self.P - self.p_init).abs().max(),
                             (self.U - self.u_init).abs().max()))

    def _conv(self, x: torch.Tensor, w: torch.Tensor, axis: int) -> torch.Tensor:
        """沿指定轴做 3-tap 一维卷积（零填充，边界规则正逆一致）。"""
        pad = self.n_taps // 2
        if axis == -1:      # 宽度方向
            w4 = w.view(1, 1, 1, -1)
            return F.conv2d(x, w4, padding=(0, pad))
        if axis == -2:      # 高度方向
            w4 = w.view(1, 1, -1, 1)
            return F.conv2d(x, w4, padding=(pad, 0))
        raise ValueError(f"不支持的轴: {axis}")

    # ---------------------------------------------------------- 一维分解/合成
    def split(self, x: torch.Tensor, axis: int):
        if axis == -1:
            return x[..., 0::2], x[..., 1::2]
        return x[..., 0::2, :], x[..., 1::2, :]

    def merge(self, even: torch.Tensor, odd: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == -1:
            n = even.shape[-1]
            out = even.new_empty(*even.shape[:-1], n * 2)
            out[..., 0::2] = even
            out[..., 1::2] = odd
        else:
            n = even.shape[-2]
            out = even.new_empty(*even.shape[:-2], n * 2, even.shape[-1])
            out[..., 0::2, :] = even
            out[..., 1::2, :] = odd
        return out

    def forward_1d(self, x: torch.Tensor, axis: int, normalize: bool = True):
        """返回 (approx, detail)。approx 对应低频，detail 对应高频。"""
        xe, xo = self.split(x, axis)
        d = xo - self._conv(xe, self.P, axis)          # 预测步
        s = xe + self._conv(d, self.U, axis)           # 更新步
        if normalize:
            s = s * math.sqrt(2.0)
            d = d / math.sqrt(2.0)
        return s, d

    def inverse_1d(self, approx: torch.Tensor, detail: torch.Tensor,
                   axis: int, normalize: bool = True) -> torch.Tensor:
        """`forward_1d` 的严格逆。"""
        s, d = approx, detail
        if normalize:
            s = s / math.sqrt(2.0)
            d = d * math.sqrt(2.0)
        # U_synth 在 synth_mismatch=0 时恒等于 U，故本改动对已有结果零影响
        xe = s - self._conv(d, self.U_synth, axis)
        xo = d + self._conv(xe, self.P, axis)
        return self.merge(xe, xo, axis)


class LiftingWavelet2D(nn.Module):
    """单级二维可分离提升小波：x → (LL, LH, HL, HH)。

    可分离实现：先沿宽度分解，再沿高度分解两个结果。
    逆变换**按相反顺序**执行（高度先合成，再宽度合成）。
    """

    def __init__(self, n_taps: int = 3, learnable: bool = True, bound: float = 0.5,
                 synth_mismatch: float = 0.0):
        super().__init__()
        self.vert = _LiftingBank1D(n_taps, learnable, bound, synth_mismatch)   # 沿高度 (dim -2)
        self.horiz = _LiftingBank1D(n_taps, learnable, bound, synth_mismatch)  # 沿宽度 (dim -1)

    def forward(self, x: torch.Tensor, normalize: bool = True):
        """子带命名与旧 `dwt_layer._haar_filters()` 对齐：

            LH = outer(lo, hi)  → 高度低通、宽度高通
            HL = outer(hi, lo)  → 高度高通、宽度低通

        故**先沿高度分解、再沿宽度分解**（顺序与直觉相反，但只有这样命名才对）。

            LL  = 高.low  ∘ 宽.low      LH = 高.low  ∘ 宽.high
            HL  = 高.high ∘ 宽.low      HH = 高.high ∘ 宽.high
        """
        Lv, Hv = self.vert.forward_1d(x, axis=-2, normalize=normalize)     # 高度
        LL, LH = self.horiz.forward_1d(Lv, axis=-1, normalize=normalize)   # 高度低通 → 宽低 LL / 宽高 LH
        HL, HH = self.horiz.forward_1d(Hv, axis=-1, normalize=normalize)   # 高度高通 → 宽低 HL / 宽高 HH
        return LL, LH, HL, HH

    def inverse(self, LL, LH, HL, HH, normalize: bool = True) -> torch.Tensor:
        """forward 的严格逆，按**相反顺序**执行（先宽度、后高度）。"""
        Lv = self.horiz.inverse_1d(LL, LH, axis=-1, normalize=normalize)
        Hv = self.horiz.inverse_1d(HL, HH, axis=-1, normalize=normalize)
        return self.vert.inverse_1d(Lv, Hv, axis=-2, normalize=normalize)


class MultiLevelLifting(nn.Module):
    """两级提升小波，产出 7 个子带，**正逆共用同一模块**。

    接口与旧实现保持一致，便于替换：
        decompose(x) -> {'LL2','LH2','HL2','HH2','LH1','HL1','HH1'}
        reconstruct(bands) -> x

    ⚠️ 与旧实现的关键区别：正变换与逆变换是**同一个对象的方法**，共享 P/U。
    这是 PR 的结构性来源——旧实现里 `MultiLevelDWT` 与 `MultiLevelIDWT`
    是两组独立参数，无法保证任何闭环性质。
    """

    # ⚠️ 2026-09-16：子带名由**硬编码 7 个（=levels 2）**改为按 levels 动态生成。
    #    此前 `--levels 3` 会 KeyError —— decompose 末尾按 BAND_NAMES 过滤时把
    #    LH3/HL3/HH3 丢掉，reconstruct 再访问就崩；而 run_ablation.ps1 里恰好
    #    写了 lvl3 作业，一跑就炸。denoiser.BANDS 是同一处硬编码，同步修。
    @staticmethod
    def band_names(levels: int) -> tuple:
        """levels 级分解的子带名：最粗的 LL，然后从高层往低层列细节。

        顺序与 decompose 的构造顺序一致，保证下游按名字取用时是稳定的。
        """
        names = [f"LL{levels}"]
        for lvl in range(levels, 0, -1):
            names += [f"LH{lvl}", f"HL{lvl}", f"HH{lvl}"]
        return tuple(names)

    def __init__(self, levels: int = 2, n_taps: int = 3, learnable: bool = True,
                 bound: float = 0.5, synth_mismatch: float = 0.0):
        super().__init__()
        self.levels = levels
        self.n_taps = n_taps
        self.bound = bound
        self.synth_mismatch = float(synth_mismatch)
        self.banks = nn.ModuleList(
            [LiftingWavelet2D(n_taps, learnable, bound, synth_mismatch)
             for _ in range(levels)]
        )

    def decompose(self, x: torch.Tensor, normalize: bool = True) -> dict:
        bands = {}
        cur = x
        per_level = []
        for lvl in range(self.levels):
            LL, LH, HL, HH = self.banks[lvl](cur, normalize=normalize)
            per_level.append((LH, HL, HH))
            cur = LL
        bands[f"LL{self.levels}"] = cur
        # 高层细节先列，然后往低层
        for lvl in range(self.levels, 0, -1):
            LH, HL, HH = per_level[lvl - 1]
            bands[f"LH{lvl}"] = LH
            bands[f"HL{lvl}"] = HL
            bands[f"HH{lvl}"] = HH
        return {k: bands[k] for k in self.band_names(self.levels)}

    def reconstruct(self, bands: dict, normalize: bool = True) -> torch.Tensor:
        cur = bands[f"LL{self.levels}"]
        for lvl in range(self.levels, 0, -1):
            suffix = str(lvl)
            cur = self.banks[lvl - 1].inverse(
                cur, bands["LH" + suffix], bands["HL" + suffix],
                bands["HH" + suffix], normalize=normalize,
            )
        return cur

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def max_drift(self) -> float:
        """所有 bank 中 taps 相对 Haar 初始值的最大偏离。"""
        return max(b.drift() for bank in self.banks for b in (bank.vert, bank.horiz))
