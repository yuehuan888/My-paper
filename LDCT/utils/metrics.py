"""
LDCT 去噪的评价指标（口径 A：RED-CNN / CTformer 谱系）。

⚠️ 口径问题比融合任务更隐蔽也更严重
------------------------------------------------------------------
2026-09-13 核查（报告：`../LDCT_HU口径核查_2026-09-13.md`）：

- **AAPM 官方从未定义任何 PSNR 口径**——官方指标是放射科医师 per-lesion /
  per-case 评分（JAFROC 仅用于破平）。
- 该数据集上**至少并存 5 个互不兼容的阵营**，同一份数据、同一个模型，
  换口径可差 **20 dB**。
- 本项目实测：同一批测试图，只换 HU 窗，identity 基线的 PSNR 从
  **27.86 dB（[-160,240]）到 46.00 dB（[-1024,3071]）**。

原因是 PSNR 的定义依赖数据范围：

    x_norm = (clip(HU, lo, hi) − lo) / R,   R = hi − lo
    PSNR = −10·log10(MSE_HU / R²) = −10·log10(MSE_HU) + 20·log10(R)

**20·log10(R) 是纯量纲偏移，与算法好坏无关。** 故"45 dB"和"30 dB"
可能只是窗不同——不写明窗的 PSNR 数字没有意义。

本模块采用的**口径 A**
------------------------------------------------------------------
被以下工作逐字沿用（已逐行核对代码）：
SSinyu/RED-CNN、SSinyu/WGAN-VGG、wdayang/CTformer (PMB 2023)、
EHSANet (2025)、MRED-Net / SDCNN (2025)。
另有 Eulig et al. (Med Phys 2024) 独立收敛到同一个 400 HU 窗（[-150,250]）。

    预处理   x = (HU + 1024) / 4096          ← 不裁剪
    评测     HU_pred = x_pred · 4096 − 1024
             HU_gt   = x_gt   · 4096 − 1024
             pred 与 gt **同时** clip 到 [-160, 240]
             PSNR = 10·log10(400² / MSE_HU)     data_range = 400
             SSIM = skimage，data_range = 400

**聚合方式**：逐切片算 PSNR/SSIM 后再平均（与 SSinyu 的 `compute_measure`
逐图调用一致）。⚠️ "先汇总 MSE 再算 PSNR"会给出不同的数——
本项目实测同一数据三种聚合跨度 0.40 dB。**必须定死一种并写进论文。**

⚠️ 冻结的窗是**单一软组织窗**。若论文需要腹部/胸部/头部各自的分窗基准
（如 Eulig 的 L/C/N 窗），本模块不适用。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 口径 A 常数
HU_OFFSET = 1024.0
HU_SCALE = 4096.0
EVAL_LO, EVAL_HI = -160.0, 240.0
DATA_RANGE = EVAL_HI - EVAL_LO          # 400

# 兼容旧名
DATA_RANGE_LEGACY = 1.0


def to_hu(x) -> np.ndarray:
    """归一化值 [0,1] -> HU。"""
    return np.asarray(x, dtype=np.float64) * HU_SCALE - HU_OFFSET


def from_hu(hu) -> np.ndarray:
    """HU -> 归一化值 [0,1]（不裁剪）。"""
    return (np.asarray(hu, dtype=np.float64) + HU_OFFSET) / HU_SCALE


def _eval_pair(pred, target):
    """把预测与参考都转到 HU 并裁剪到评测窗。二者用**同一个**窗。"""
    p = to_hu(pred)
    g = to_hu(target)
    if p.shape != g.shape:
        raise ValueError(f"尺寸不一致: {p.shape} vs {g.shape}")
    return np.clip(p, EVAL_LO, EVAL_HI), np.clip(g, EVAL_LO, EVAL_HI)


def psnr(pred, target) -> float:
    """口径 A 的 PSNR (dB)。

    `pred` / `target` 是 [0,1] 域的归一化值（网络输入/输出格式）。
    内部先转 HU、clip 到 [-160,240]，再按 data_range=400 计算。
    完全相同返回 inf。
    """
    p, g = _eval_pair(pred, target)
    mse = float(np.mean((p - g) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(DATA_RANGE ** 2 / mse))


_SSIM_WINDOW = None


def _ssim_window(n: int = 11, sigma: float = 1.5) -> np.ndarray:
    """pytorch-ssim 的高斯窗：gaussian(11,1.5) 的外积，归一化。"""
    global _SSIM_WINDOW
    if _SSIM_WINDOW is None:
        x = np.arange(n) - n // 2
        g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        _SSIM_WINDOW = np.outer(g, g)
    return _SSIM_WINDOW


def ssim(pred, target) -> float:
    """口径 A 的 SSIM，data_range=400。

    **实现遵循 SSinyu/RED-CNN 的 `measure.py`**（源自 `pytorch-ssim`），
    而不是 skimage 的 `structural_similarity`——两者不等价：

    | | 参考实现（本函数） | skimage |
    |---|---|---|
    | 填充 | **零填充**（`F.conv2d(padding=5)`） | 反射/有效区域裁剪 |
    | 聚合 | 对 **全图** `ssim_map` 取均值 | 只对有效区域取均值 |
    | 方差 | `E[x²] − E[x]²`（总体） | 同（`use_sample_covariance=False`） |

    实测差异（L506，211 片）：参考实现 0.8759 vs skimage(gaussian,11,1.5) 0.8710。

    ⚠️ 已知偏差：本轮调研报告给出该量为 0.8838，与本实现差 0.0079。
    本实现已与参考仓库代码**逐字对照**（torch 版与 numpy 移植版结果一致，
    均为 0.8759），故以本实现为准；PSNR 三项数值则与调研报告**精确吻合**。
    """
    from scipy.ndimage import convolve

    p, g = _eval_pair(pred, target)
    if min(p.shape) < 11:
        raise ValueError(f"SSIM 局部窗口需边长 >= 11，实际 {p.shape}")
    w = _ssim_window()
    c = lambda a: convolve(a, w, mode="constant", cval=0.0)   # 零填充
    mu1, mu2 = c(g), c(p)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = c(g * g) - mu1_sq
    s2 = c(p * p) - mu2_sq
    s12 = c(g * p) - mu1_mu2
    C1, C2 = (0.01 * DATA_RANGE) ** 2, (0.03 * DATA_RANGE) ** 2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (s1 + s2 + C2))
    return float(m.mean())


def evaluate(pred, target) -> dict:
    return {"PSNR": psnr(pred, target), "SSIM": ssim(pred, target)}


def mean_metrics(rows) -> dict:
    """逐切片指标的平均。

    ⚠️ 口径 A 要求**逐切片算指标后平均**（与 SSinyu 的逐图调用一致），
    而不是先把所有切片的 MSE 汇总再算一次 PSNR。两者不等价
    （本项目实测跨度 0.40 dB）。
    """
    return {
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in rows])),
    }


__all__ = ["psnr", "ssim", "evaluate", "mean_metrics",
           "to_hu", "from_hu",
           "HU_OFFSET", "HU_SCALE", "EVAL_LO", "EVAL_HI", "DATA_RANGE"]
