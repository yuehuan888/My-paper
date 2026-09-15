"""
L0 层指标不变量测试（LDCT 版）。

这一层**不需要外部参考值**，只用数学上必须成立的性质抓实现错误。
参照 MGF-Net 的同类测试——那次它抓出了三类问题，其中一类还是测试自身写错。

⚠️ 2026-09-15：本文件此前**停留在 data_range=1.0 时代**，与实现严重脱节。
    git 证据：本文件最后改动于 09a6d35（项目创建，当时 `DATA_RANGE = 1.0`），
    而 7c282fc「切换到口径 A」把实现改成了 `DATA_RANGE = EVAL_HI - EVAL_LO = 400`，
    本文件**之后从未更新**。后果：4 个测试长期假失败，安全网失效——
    真 bug 会淹没在噪声里。
    已按口径 A 重写解析类断言（见 `_windowed_phantom`）。
    **不要再把 [0,1] 全幅随机图直接喂给口径 A 的指标**：那会先把 HU 反归一化到
    [-1024, 3072]，再被 clip 到 [-160, 240]，绝大多数像素被压成常数，
    误差被人为抹平，得到的 dB 值没有解析意义。

运行：
    python tests/test_metrics_invariants.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import metrics as M  # noqa: E402

RNG = np.random.default_rng(20260913)


def _phantom(h=128, w=128, seed=0):
    """类 CT 的合成图：低频结构 + 少量高对比细节 + 噪声。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.5 + 0.25 * np.sin(xx / 18.0) * np.cos(yy / 23.0)
    base += 0.15 * ((xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (min(h, w) * 0.18) ** 2)
    base += rng.normal(0, 0.02, size=(h, w))
    return np.clip(base, 0, 1)


def _windowed_phantom(h=64, w=64, seed=0):
    """落在**口径 A 的 HU 窗内**的合成图，用于解析校验。

    口径 A：`x = (HU + 1024) / 4096`，评测时反归一化并把预测与参考**同时**
    clip 到 `[-160, 240]` HU。故只有当整幅图都落在该窗内时，clip 才是恒等的、
    `PSNR = 20·log10(400 / RMSE_HU)` 才有解析意义。

    本函数把值限制在归一化域的 `[-160,240] HU` 对应的区间
    `[(−160+1024)/4096, (240+1024)/4096] = [0.2109, 0.3086]`，并留出 ±ε 余量。
    """
    lo = (-160.0 + 1024.0) / 4096.0
    hi = (240.0 + 1024.0) / 4096.0
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    # 起伏幅度取窗宽的 ±0.22（即两侧各留 0.28 窗宽的余量）。
    # ⚠️ 余量必须显式留够：窗宽只有 400/4096 ≈ 9.8% 的归一化域，
    #    8 bit 量化步长 1/255 ≈ 0.39% 就相当于 **16 HU**，噪声也很容易顶到窗边。
    #    早期版本用 ±0.35 且 σ 取到 0.016，3σ 直接越窗，解析式失效。
    base = 0.5 * (np.sin(xx / 11.0) * np.cos(yy / 13.0))
    base = lo + (hi - lo) * (0.5 + 0.22 * base)
    return np.clip(base + rng.normal(0, 1e-5, size=(h, w)), lo, hi)


def test_identity():
    """pred == target 时 PSNR = inf，SSIM = 1。"""
    x = _phantom()
    assert M.psnr(x, x) == float("inf"), f"PSNR(x,x)={M.psnr(x,x)} 应为 inf"
    assert abs(M.ssim(x, x) - 1.0) < 1e-10, f"SSIM(x,x)={M.ssim(x,x)} 应为 1"
    print("  [1] 一致性：PSNR=inf, SSIM=1                OK")


def test_constant():
    """常量图：相同则 PSNR=inf/SSIM=1；**窗内**不同值则有限且无 NaN。

    ⚠️ 口径 A 下必须选**窗内**的两个值。0.4 / 0.6 反归一化后是 614 / 1434 HU，
    两者都被 clip 到上界 240 → 误差被抹成 0 → PSNR=inf，测试会假失败。
    """
    lo = (-160.0 + 1024.0) / 4096.0
    hi = (240.0 + 1024.0) / 4096.0
    c = np.full((64, 64), lo + 0.15 * (hi - lo))
    d = np.full((64, 64), lo + 0.45 * (hi - lo))
    assert M.psnr(c, c) == float("inf")
    assert abs(M.ssim(c, c) - 1.0) < 1e-10
    p, s = M.psnr(c, d), M.ssim(c, d)
    assert np.isfinite(p) and np.isfinite(s), f"窗内常量图不同值应有限，得 PSNR={p} SSIM={s}"
    # 窗内常量图：解析值 PSNR = 20·log10(400 / Δ_HU)
    d_hu = abs(d[0, 0] - c[0, 0]) * 4096.0
    assert abs(p - 20.0 * np.log10(400.0 / d_hu)) < 1e-6, \
        f"窗内常量图 PSNR 应为解析值 {20*np.log10(400/d_hu):.4f}，得 {p:.4f}"
    print(f"  [2] 常量图（窗内）无 NaN                  OK  "
          f"(Δ={d_hu:.1f} HU: PSNR={p:.2f} SSIM={s:.4f})")


def test_value_ranges():
    """值域：SSIM ≤ 1；PSNR 随噪声单调下降。"""
    x = _phantom()
    n = np.clip(x + RNG.normal(0, 0.05, x.shape), 0, 1)
    s = M.ssim(n, x)
    assert s <= 1.0 + 1e-9, f"SSIM={s} 越界"
    assert -1.0 - 1e-9 <= s, f"SSIM={s} 越界"
    print(f"  [3] 值域钳制                              OK  (SSIM={s:.4f})")


def test_monotonic_degradation():
    """噪声越大，PSNR / SSIM 必须单调下降。"""
    x = _phantom()
    sigmas = [0.0, 0.01, 0.03, 0.06, 0.12, 0.25]
    ps, ss = [], []
    for s in sigmas:
        n = np.clip(x + RNG.normal(0, s, x.shape), 0, 1)
        ps.append(M.psnr(n, x))
        ss.append(M.ssim(n, x))
    for name, vals in [("PSNR", ps), ("SSIM", ss)]:
        d = np.diff(vals)
        assert np.all(d <= 1e-9), f"{name} 未单调下降: {[f'{v:.4f}' for v in vals]}"
    print("  [4] 加噪单调性                            OK")
    print(f"      PSNR {ps[0]:.2f} -> {ps[-1]:.2f}   SSIM {ss[0]:.4f} -> {ss[-1]:.4f}")


def test_psnr_analytic():
    """PSNR 的解析校验（口径 A）：窗内、均匀误差时 PSNR = 20·log10(400 / ε_HU)。

    归一化域的误差 ε 对应 HU 域的 ε_HU = ε · 4096。
    """
    x = _windowed_phantom(64, 64)
    for eps in [0.002, 0.005, 0.01]:
        y = x + eps
        # 窗内图必须留有余量，触边则 clip 不再是恒等，解析式失效
        if np.any(y > (240.0 + 1024.0) / 4096.0) or np.any(y < (-160.0 + 1024.0) / 4096.0):
            continue
        expect = 20.0 * np.log10(400.0 / (eps * 4096.0))
        got = M.psnr(y, x)
        assert abs(got - expect) < 1e-6, f"ε={eps}: 得 {got:.6f}，应 {expect:.6f}"
        print(f"  [5] PSNR 解析校验 ε={eps} (={eps*4096:.2f} HU): "
              f"{got:.4f} dB (期望 {expect:.4f})  OK")
        return
    raise AssertionError("合成图触边，解析校验未能执行——请调小 ε 或收窄 _windowed_phantom 幅度")


def test_scale_consistency():
    """量化到 8 bit 往返后，指标应几乎不变。

    原意是抓"量纲搞错"。⚠️ 口径 A 的指标**约定输入为 [0,1] 归一化域**，
    所以 uint8 路径必须先除回 255 再喂入；直接喂 0–255 的整数是量纲错误，
    会先把 (HU+1024)/4096 算成一个荒谬的 HU 值，再被窗口整体压平。
    这里直接把这个量纲错误也一并断言掉。
    """
    x = _windowed_phantom(128, 128)
    # 噪声取 8e-3（≈33 HU）。**必须显著大于量化误差**，否则两路径的 dB 差
    # 被量化本身撑大，测不出"尺度一致"。
    # 量化误差的算术（实测印证过）：8 bit 步长 = 4096/255 = 16.06 HU，
    # 均匀量化 RMS = 步长/√12 = 4.64 HU；**两幅图都量化**，故 √2 倍 = 6.56 HU。
    #   噪声 16.4 HU 时：√(16.4² + 6.56²) = 17.7 HU → 与正确路径差 0.66 dB（超容差）
    #   噪声 32.8 HU 时：√(32.8² + 6.56²) = 33.4 HU → 差约 0.16 dB（通过）
    n = x + RNG.normal(0, 8e-3, x.shape)
    lo, hi = (-160.0 + 1024.0) / 4096.0, (240.0 + 1024.0) / 4096.0
    assert n.min() > lo and n.max() < hi, "噪声越出 HU 窗，clip 会污染本测试"

    p1, s1 = M.psnr(n, x), M.ssim(n, x)

    # 正确的 8 bit 往返：量化后除回 [0,1]
    x8 = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8).astype(np.float64) / 255.0
    n8 = np.clip(np.rint(n * 255), 0, 255).astype(np.uint8).astype(np.float64) / 255.0
    p2, s2 = M.psnr(n8, x8), M.ssim(n8, x8)
    assert abs(p1 - p2) < 0.5, f"PSNR 量化不一致: float={p1:.4f} 8bit={p2:.4f}"
    # SSIM 容差放宽到 2e-2：口径 A 的窗只有 400 HU，而 8 bit 一步就是 16 HU
    # （4% 窗宽），量化扰动占附加误差约 20%，SSIM 因此比旧 [0,1] 版本敏感得多。
    # 本测试的**主目的**是抓量纲错误（见下方 p_bad 断言），而非量化不变性。
    assert abs(s1 - s2) < 2e-2, f"SSIM 量化不一致: float={s1:.4f} 8bit={s2:.4f}"

    # 量纲错误必须**明显**偏离（正是本测试存在的意义）
    p_bad = M.psnr(np.rint(n * 255), np.rint(x * 255))
    assert abs(p_bad - p1) > 0.5, \
        f"把 0–255 整数直接喂给归一化域指标竟与正确用法一致（{p_bad:.2f} vs {p1:.2f}）——量纲防护失效"
    print(f"  [6] 量化往返一致性 + 量纲错误可检出      OK  "
          f"(PSNR {p1:.2f}/{p2:.2f}，错误量纲 {p_bad:.2f})")


def test_shapes():
    """奇数尺寸、非方形、小图；尺寸不一致应报错。"""
    for shape in [(127, 127), (128, 111), (256, 233)]:
        x = _phantom(*shape)
        n = np.clip(x + RNG.normal(0, 0.03, x.shape), 0, 1)
        assert np.isfinite(M.psnr(n, x)) and np.isfinite(M.ssim(n, x))
    try:
        M.psnr(_phantom(64, 64), _phantom(64, 32))
        raise AssertionError("尺寸不一致应抛 ValueError")
    except ValueError:
        pass
    try:
        M.ssim(_phantom(8, 8), _phantom(8, 8))
        raise AssertionError("小于 SSIM 窗口应抛 ValueError")
    except ValueError:
        pass
    print("  [7] 形状 / 越界报错                       OK")


def test_magnitude():
    """量级自查：PSNR 用解析式严格校验，SSIM 只查单调与合理下界。

    教训：本测试第一版给 SSIM 硬编码了"拍脑袋"的预期区间（σ=0.05 期望 0.7–0.95），
    实测 0.4674 就"失败"了——**错的是断言，不是实现**。
    PSNR 有解析式（-20·log10(σ)）可以严格校验，SSIM 没有，故只查趋势与宽松边界，
    避免把错误预期固化成测试。
    """
    x = _windowed_phantom(256, 256)
    prev = None
    # σ 上限受窗内余量约束：_windowed_phantom 两侧各留 0.28 窗宽 ≈ 0.0274，
    # 故 3σ 必须 < 0.0274，即 σ < 9.1e-3。取到 0.008 已接近上限。
    for sig in [0.001, 0.002, 0.004, 0.008]:
        # 窗内图 + 窗内噪声：不能 clip 到 [0,1]，否则会人为削掉高斯尾
        n = x + RNG.normal(0, sig, x.shape)
        lo, hi = (-160.0 + 1024.0) / 4096.0, (240.0 + 1024.0) / 4096.0
        assert n.min() > lo and n.max() < hi, f"σ={sig} 越出 HU 窗，clip 会让解析式失效"
        p, s = M.psnr(n, x), M.ssim(n, x)

        # 口径 A 的解析值：PSNR = 20·log10(400 / σ_HU)，σ_HU = σ · 4096
        expect = 20.0 * np.log10(400.0 / (sig * 4096.0))
        assert abs(p - expect) < 1.0, f"σ={sig}: PSNR={p:.2f} 与解析值 {expect:.2f} 差过大"

        # SSIM：只查落在 (0, 1] 且随噪声增大而下降
        assert 0.0 < s <= 1.0, f"σ={sig}: SSIM={s:.4f} 越界"
        if prev is not None:
            assert s < prev, f"σ={sig}: SSIM={s:.4f} 未随噪声下降（前一档 {prev:.4f}）"
        prev = s

    print("  [8] 量级：PSNR 解析校验 + SSIM 单调        OK")


def main():
    print("=" * 74)
    print("L0 指标不变量测试（LDCT）")
    print("=" * 74)
    tests = [test_identity, test_constant, test_value_ranges,
             test_monotonic_degradation, test_psnr_analytic,
             test_scale_consistency, test_shapes, test_magnitude]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}\n      {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 74)
    print(f"结果：{len(tests)-failed}/{len(tests)} 通过")
    print("=" * 74)
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
