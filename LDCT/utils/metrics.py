"""
LDCT 去噪的评价指标：PSNR 与 SSIM。

与图像融合任务的关键区别
------------------------------------------------------------------
融合的指标是**源参考型**（与两个源图比），无唯一真值，可靠"贴近单一源"刷高
（实测：MI 的最优解是 α=0.10、VIF 的最优解是纯 MRI，见 MGF-Net 的教训）。

本任务的指标是**真值型**：输出直接与同患者的全剂量 CT 逐像素比较，
不存在口径争议或刷分空间。**这是换方向的主要理由之一。**

口径约定（写进论文时必须说明）
------------------------------------------------------------------
1. **计算域 [0, 1]**：LoDoPaB 的图像即在此范围（362×362, float32）。
   `data_range=1.0`。这与 MGF-Net 的 [0,255] 域不同，不要沿用那边的常数。
2. **PSNR**：`10·log10(data_range² / MSE)`。MSE 逐像素、全图平均。
   完全相同时返回 inf。
3. **SSIM**：局部窗口，高斯窗 11×11、σ=1.5、`use_sample_covariance=False`。
   与 skimage 的显式参数一致（其默认参数与 Wang 的 MATLAB 实现不符）。
4. **不做裁剪**：不把预测值 clip 到 [0,1] 后再算指标——那会掩盖模型的越界行为。
   若官方评测有裁剪，需在文档中明确并保持一致。

⚠️ 待办：与 `dival` 的官方实现对照（见 tasks #19）。
   本实现与官方若有差异，以此处的口径声明为准进行调整。
"""

from __future__ import annotations

import numpy as np

DATA_RANGE = 1.0


def _as_float(img) -> np.ndarray:
    """统一为 float64 的 numpy 数组。不做值域变换（本任务输入本就是 [0,1]）。"""
    a = np.asarray(img)
    if a.dtype == np.uint8:
        # 若误传 8-bit 图，按 [0,255] 域转回 [0,1]，并在使用处提示
        return a.astype(np.float64) / 255.0
    return a.astype(np.float64)


def psnr(pred, target, data_range: float = DATA_RANGE) -> float:
    """峰值信噪比 (dB)。完全相同返回 inf。"""
    a, b = _as_float(pred), _as_float(target)
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def ssim(pred, target, data_range: float = DATA_RANGE) -> float:
    """结构相似性指数，局部窗口（高斯 11×11, σ=1.5）。"""
    from skimage.metrics import structural_similarity

    a, b = _as_float(pred), _as_float(target)
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a.shape} vs {b.shape}")
    if min(a.shape) < 11:
        raise ValueError(f"SSIM 局部窗口需边长 >= 11，实际 {a.shape}")
    return float(structural_similarity(
        a, b, data_range=data_range,
        gaussian_weights=True, sigma=1.5,
        use_sample_covariance=False, win_size=11,
    ))


def evaluate(pred, target) -> dict:
    """返回 {'PSNR': ..., 'SSIM': ...}。"""
    return {"PSNR": psnr(pred, target), "SSIM": ssim(pred, target)}


def mean_metrics(rows) -> dict:
    """对逐样本结果取均值（无特殊聚合规则——本任务指标无歧义）。"""
    return {
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in rows])),
    }


__all__ = ["psnr", "ssim", "evaluate", "mean_metrics", "DATA_RANGE"]
