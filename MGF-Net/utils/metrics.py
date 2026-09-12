"""
医学图像融合的标准评价指标。

设计约定（非常重要，改动前请先读）
------------------------------------------------------------------
本项目所有指标实现共享以下约定，违反任一条都会导致与其他论文的数字不可比：

1. **计算域固定为 [0, 255]**。SD / SF / PSNR / SSIM / VIF 的常数与阈值
   全部绑定这个量纲。输入若为 float 且落在 [0, 1]，会被自动 ×255。
   （实测教训：同一张图在 [0,1] 域算 SD 得 73.86，×255 后变 18834，两者差 255 倍。）

2. **输入接受两种形式**：uint8、或 float 且值域为 [0, 1]。
   两种形式喂入同一张图，结果必须完全相同（见 tests/test_metrics_invariants.py）。
   不接受值域为 [0, 255] 的 float——那会与 [0,1] 的 float 产生歧义。

3. **灰度**：本项目数据本身就是灰度 PNG。若将来支持彩色，须用
   YCbCr 的 Y 通道（`cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[..., 0]`），
   而不是简单加权平均——VIF-Benchmark 的 RGB2Gray.m 即用 Y。

4. **融合指标的两源聚合方式**在本文件中逐个显式定义，不依赖默认行为。
   MI 用**求和**（Qu et al. 2002 原式），CC / PSNR / SSIM / VIF 用**均值**。
   求和与均值差 2 倍且都"看似合理"，故必须写进论文。

5. **数值边界**：平坦区的 0/0 必须显式处理，否则 NaN 会沿 sum 传播成全图 NaN。
   常量图测试是必炸项（见 L0 测试）。

参考
------------------------------------------------------------------
- EN / SD / SF：融合文献通用定义
- PSNR / CC / SSIM：Wang et al. 2004, IEEE TIP
- MI：Qu, Zhang, Yan 2002, Information Fusion
- SCD：Aslantas & Bendes 2015（**原论文 Eq.10 分母漏印根号，勿照抄**）
- Qabf：Xydeas & Petrovic 2000, Electronics Letters
- VIF-P：Sheikh & Bovik 2006, IEEE TIP（像素域，非融合专用的 VIFF）
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 常数
DOMAIN_MAX = 255.0

# Qabf 常数，取自 Xydeas & Petrovic 2000 原文
_QABF_L = 1.0
_QABF_TG = 0.9994
_QABF_KG = -15.0
_QABF_DG = 0.5
_QABF_TA = 0.9879
_QABF_KA = -22.0
_QABF_DA = 0.8

_EPS = np.finfo(np.float64).eps

_SOBEL_X = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


# ------------------------------------------------------------------ 输入处理
def as_255(img: np.ndarray) -> np.ndarray:
    """把输入统一为 float64 且值域 [0, 255]。

    uint8            -> 直接转 float64
    float in [0,1]   -> ×255
    其他             -> ValueError（不做猜测，避免与 [0,255] 的 float 歧义）
    """
    a = np.asarray(img)
    if a.dtype == np.uint8:
        return a.astype(np.float64)
    if np.issubdtype(a.dtype, np.floating):
        lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
        if hi <= 1.0 + 1e-9 and lo >= -1e-9:
            return a.astype(np.float64) * DOMAIN_MAX
        raise ValueError(
            f"float 输入的值域应为 [0,1]，实际为 [{lo:.4g}, {hi:.4g}]。"
            "若这是 [0,255] 的 float，请先自行 /255 或用 uint8 传入。"
        )
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(np.float64)
    raise ValueError(f"不支持的 dtype: {a.dtype}")


def _as_raw(img: np.ndarray) -> np.ndarray:
    """只做 dtype 转换，不做值域校验、不做缩放。

    用于**尺度无关**的指标（Pearson 相关系数），以及所有已经被
    `as_255` 归一化过的内部数据——避免二次缩放。

    之所以需要它：SCD 会计算 `F - S1` 这类差分图，值域含负数；
    若走 `as_255` 会被值域校验拒绝（这是设计上的自我保护，
    但同时暴露了"公开函数被内部重复调用"的问题）。
    """
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float64)
    raise ValueError(f"不支持的 dtype: {a.dtype}")


def _as_u8(img: np.ndarray) -> np.ndarray:
    """转为 uint8 用于直方图类计算（MI / EN）。"""
    return np.clip(np.rint(as_255(img)), 0, 255).astype(np.uint8)


def _check_same_shape(*imgs: np.ndarray) -> None:
    shapes = {np.asarray(i).shape for i in imgs}
    if len(shapes) != 1:
        raise ValueError(f"图像尺寸不一致: {sorted(shapes)}")


# ------------------------------------------------------------------ 单图指标
def entropy(img: np.ndarray) -> float:
    """EN 信息熵，单位 bit。

    陷阱：`np.histogram` 默认 bins=10，会**静默**给出约 3.3 的假值
    （正确值应在 6~8）。必须显式 bins=256 且用 log2。
    log 用 e 会让结果差 0.693 倍。
    """
    a = _as_u8(img)
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    nz = p > 0
    return float(-np.sum(p[nz] * np.log2(p[nz])))


def std_dev(img: np.ndarray) -> float:
    """SD 标准差，[0,255] 域。

    ddof=1（样本标准差）以对齐 MATLAB `std` 的默认行为；
    大图上 ddof=0 与 ddof=1 相差约 1.0000077 倍，可忽略但需一致。
    """
    return float(np.std(as_255(img), ddof=1))


def spatial_frequency(img: np.ndarray) -> float:
    """SF 空间频率，[0,255] 域。

    定义：
        RF = sqrt( mean_ij (F[i,j] - F[i,j-1])^2 )
        CF = sqrt( mean_ij (F[i,j] - F[i-1,j])^2 )
        SF = sqrt(RF^2 + CF^2)

    分母取 M*N（即差分数组大小），与文献常见写法一致。
    边界用前向差分，**不要用 np.roll**——那会在边界造出假的高频。
    """
    a = as_255(img)
    df_r = np.diff(a, axis=1)          # 水平方向差分, shape (M, N-1)
    df_c = np.diff(a, axis=0)          # 垂直方向差分, shape (M-1, N)
    rf = np.sqrt(np.mean(df_r ** 2)) if df_r.size else 0.0
    cf = np.sqrt(np.mean(df_c ** 2)) if df_c.size else 0.0
    return float(np.sqrt(rf ** 2 + cf ** 2))


# ------------------------------------------------------------------ 两图指标
def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """PSNR (dB)，峰值 255。

    陷阱：该系列代码通病是 MSE 在 [0,1] 尺度算却除以 255 取对数，
    导致固定虚高 20*log10(255) = 48.13 dB。本实现统一在 [0,255] 域算。
    完全相同时返回 inf（不是 0，也不是某个大数）。
    """
    a, b = as_255(img1), as_255(img2)
    _check_same_shape(a, b)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(20.0 * np.log10(DOMAIN_MAX / np.sqrt(mse)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """纯 Pearson 相关系数，作用于任意值域的原始数组。"""
    _check_same_shape(a, b)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    if denom < _EPS:
        return 0.0
    return float(np.sum(a * b) / denom)


def correlation_coefficient(img1: np.ndarray, img2: np.ndarray) -> float:
    """CC Pearson 相关系数，值域 [-1, 1]。

    必须去均值且分母带根号。注意 sewar.scc 是拉普拉斯高通变体，
    不是标准 CC，不可混用。

    CC 对仿射变换不变，**不走 [0,255] 归一化**——这既让 uint8 与
    float[0,1] 输入天然一致，也允许 SCD 传入含负值的差分图。
    """
    return _pearson(_as_raw(img1), _as_raw(img2))


def mutual_information(img1: np.ndarray, img2: np.ndarray, bins: int = 256) -> float:
    """MI 互信息（bit），用 256×256 联合直方图。

    陷阱：整型输入。浮点会退化成 65536 bin 的直方图，结果不可比。
    """
    a, b = _as_u8(img1), _as_u8(img2)
    _check_same_shape(a, b)
    joint = np.histogram2d(
        a.ravel(), b.ravel(), bins=bins, range=[[0, 256], [0, 256]]
    )[0]
    total = joint.sum()
    if total == 0:
        return 0.0
    joint = joint / total
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    outer = np.outer(pa, pb)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log2(joint[nz] / outer[nz])))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """SSIM，局部窗口（高斯窗 11×11, σ=1.5），[0,255] 域。

    用 skimage 的实现并显式指定参数——skimage 的默认值
    （win_size=7 + uniform_filter + 样本协方差）与 Wang 的 MATLAB `ssim.m`
    不一致，会差在千分位。

    ⚠️ 本文件原先的实现是**全图统计**（窗口=整张图），那不是"变体"而是错。
    """
    from skimage.metrics import structural_similarity

    a, b = as_255(img1), as_255(img2)
    _check_same_shape(a, b)
    if min(a.shape) < 11:
        raise ValueError(f"SSIM 局部窗口需图像边长 >= 11，实际 {a.shape}")
    return float(
        structural_similarity(
            a,
            b,
            data_range=DOMAIN_MAX,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            win_size=11,
        )
    )


# ------------------------------------------------------------------ 融合专用
def scd(fused: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """SCD 差异相关和。

        SCD = corr(F - S2, S1) + corr(F - S1, S2)

    其中 corr 为标准 Pearson 相关系数。

    ⚠️ Aslantas & Bendes 2015 原论文 Eq.10 **分母漏印了根号**，照抄会得到
    远超 1 的值。合理区间约 1.3~1.9；**> 2 基本判定为 bug**
    （两个 Pearson 各 <= 1，理论最大为 2）。
    """
    f, a, b = as_255(fused), as_255(s1), as_255(s2)
    _check_same_shape(f, a, b)
    # 用 _pearson 而非 correlation_coefficient：差分图已在本函数内归一化过，
    # 再走一次 as_255 会因值为负数而触发值域校验报错。
    return float(_pearson(f - b, a) + _pearson(f - a, b))


def _sobel_mag_ori(img: np.ndarray):
    """返回 (梯度幅值, 梯度方向)。方向用 atan2(gy, gx)，值域 (-pi, pi]。"""
    import cv2

    gx = cv2.filter2D(img, cv2.CV_64F, _SOBEL_X, borderType=cv2.BORDER_REPLICATE)
    gy = cv2.filter2D(img, cv2.CV_64F, _SOBEL_Y, borderType=cv2.BORDER_REPLICATE)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx)
    return mag, ori


def _qabf_one(fused, src):
    """单个源图对融合图的 Qabf 贡献。返回 (加权和, 权重和)。"""
    gf, af = _sobel_mag_ori(fused)
    gs, as_ = _sobel_mag_ori(src)

    # --- 边缘强度保持度 Qg ---
    # G^AF = min/max，即始终 <= 1
    denom = np.maximum(gs, gf)
    numer = np.minimum(gs, gf)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, numer / denom, 1.0)
    # 两图都平坦时视为完全保持（1.0），否则 0/0 会产生 NaN 并传播
    ratio = np.where(denom > 0, ratio, 1.0)
    qg = _QABF_TG / (1.0 + np.exp(_QABF_KG * (ratio - _QABF_DG)))

    # --- 边缘方向保持度 Qa ---
    # 方向差折叠到 [0, pi/2]
    d_ang = np.abs(as_ - af)
    d_ang = np.abs(((d_ang + np.pi / 2) % np.pi) - np.pi / 2)
    a_pres = 1.0 - d_ang / (np.pi / 2)
    qa = _QABF_TA / (1.0 + np.exp(_QABF_KA * (a_pres - _QABF_DA)))

    w = gs ** _QABF_L
    q = np.nan_to_num(qg * qa * w, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(q)), float(np.sum(w))


def qabf(fused: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> float:
    """Qabf 边缘保持度，值域 [0, 1]。Xydeas & Petrovic 2000。

    陷阱：平坦区的 0/0 必须显式处理，否则 NaN 会沿 sum 传播成全图 NaN。
    常量图输入是必测项。
    """
    f, a, b = as_255(fused), as_255(s1), as_255(s2)
    _check_same_shape(f, a, b)
    num1, w1 = _qabf_one(f, a)
    num2, w2 = _qabf_one(f, b)
    wsum = w1 + w2
    if wsum < _EPS:
        # 两源都完全平坦：无边缘可保留，按惯例返回 1.0（完全保持）
        return 1.0
    return float((num1 + num2) / wsum)


# ------------------------------------------------------------------ VIF-P
def _gaussian_kernel(n: int, sigma: float) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _filter_2d(img: np.ndarray, n: int, sigma: float) -> np.ndarray:
    import cv2

    k = _gaussian_kernel(n, sigma)
    out = cv2.sepFilter2D(img, cv2.CV_64F, k, k, borderType=cv2.BORDER_REPLICATE)
    return out


def _downsample(img: np.ndarray) -> np.ndarray:
    """低通后 2 倍下采样（取偶数行/列）。"""
    return img[::2, ::2]


def vifp(img1: np.ndarray, img2: np.ndarray, sigma_nsq: float = 2.0) -> float:
    """VIF-P，Sheikh & Bovik 像素域视觉信息保真度，4 尺度。

    ⚠️ 这是 VIF-P，**不是** Han et al. 2013 的融合专用 VIFF（ComVidVindG）。
    二者不能互换署名。若论文要报 VIFF，须另行说明定义来源。

    参考：vifp_mscale.m (Sheikh & Bovik, UT Austin)，该文件有明确再分发授权，
    本实现为按算法重写。

    sigma_nsq=2 绑定 [0,255] 域——这是必须固定计算域的原因之一。
    """
    ref, dist = as_255(img1), as_255(img2)
    _check_same_shape(ref, dist)
    if min(ref.shape) < 16:
        raise ValueError(f"VIF-P 需图像边长 >= 16，实际 {ref.shape}")

    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        n = int(2 ** (4 - scale + 1) + 1)
        sd = n / 5.0

        if scale > 1:
            ref = _downsample(_filter_2d(ref, n, sd))
            dist = _downsample(_filter_2d(dist, n, sd))

        mu1 = _filter_2d(ref, n, sd)
        mu2 = _filter_2d(dist, n, sd)

        sigma1_sq = _filter_2d(ref * ref, n, sd) - mu1 ** 2
        sigma2_sq = _filter_2d(dist * dist, n, sd) - mu2 ** 2
        sigma12 = _filter_2d(ref * dist, n, sd) - mu1 * mu2

        sigma1_sq = np.maximum(sigma1_sq, 0.0)
        sigma2_sq = np.maximum(sigma2_sq, 0.0)

        g = sigma12 / (sigma1_sq + _EPS)
        sv_sq = sigma2_sq - g * sigma12

        mask = sigma1_sq < _EPS
        g = np.where(mask, 0.0, g)
        sv_sq = np.where(mask, sigma2_sq, sv_sq)
        sigma1_sq = np.where(mask, 0.0, sigma1_sq)

        mask = sigma2_sq < _EPS
        g = np.where(mask, 0.0, g)
        sv_sq = np.where(mask, 0.0, sv_sq)

        mask = g < 0
        sv_sq = np.where(mask, sigma2_sq, sv_sq)
        g = np.where(mask, 0.0, g)

        sv_sq = np.maximum(sv_sq, _EPS)

        num += float(np.sum(np.log10(1.0 + (g ** 2) * sigma1_sq / (sv_sq + sigma_nsq))))
        den += float(np.sum(np.log10(1.0 + sigma1_sq / sigma_nsq)))

    if den < _EPS:
        return 0.0
    return float(num / den)


# ------------------------------------------------------------------ 聚合
def evaluate_pair(fused: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> dict:
    """对一对源图与融合图计算全部指标。

    两源聚合方式（写进论文时必须说明）：
        MI              : 求和   MI(F,S1) + MI(F,S2)          [Qu et al. 2002]
        CC / PSNR / SSIM / VIF : 均值   0.5 * (·(F,S1) + ·(F,S2))

    例外：PSNR 若任一源为 inf（完全相同时），均值仍为 inf。
    """
    # 注意：这里**不能**预先 as_255 再传给下面各函数——那些函数内部会各自
    # 归一化，重复转换会把已经是 [0,255] 的 float 判为非法值域。直接传原始输入。
    _check_same_shape(fused, s1, s2)

    mi_1, mi_2 = mutual_information(fused, s1), mutual_information(fused, s2)
    cc_1, cc_2 = (correlation_coefficient(fused, s1),
                  correlation_coefficient(fused, s2))
    ps_1, ps_2 = psnr(fused, s1), psnr(fused, s2)
    ss_1, ss_2 = ssim(fused, s1), ssim(fused, s2)
    vi_1, vi_2 = vifp(fused, s1), vifp(fused, s2)

    return {
        "EN": entropy(fused),
        "SD": std_dev(fused),
        "SF": spatial_frequency(fused),
        "MI": mi_1 + mi_2,
        "CC": 0.5 * (cc_1 + cc_2),
        "PSNR": 0.5 * (ps_1 + ps_2),
        "SSIM": 0.5 * (ss_1 + ss_2),
        "VIF": 0.5 * (vi_1 + vi_2),
        "SCD": scd(fused, s1, s2),
        "Qabf": qabf(fused, s1, s2),
        # 分项，便于排查
        "_SSIM_S1": ss_1,
        "_SSIM_S2": ss_2,
        "_MI_S1": mi_1,
        "_MI_S2": mi_2,
    }


__all__ = [
    "as_255",
    "entropy",
    "std_dev",
    "spatial_frequency",
    "psnr",
    "correlation_coefficient",
    "mutual_information",
    "ssim",
    "scd",
    "qabf",
    "vifp",
    "evaluate_pair",
]
