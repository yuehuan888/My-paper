"""
L0 层指标不变量测试（LDCT 版）。

这一层**不需要外部参考值**，只用数学上必须成立的性质抓实现错误。
参照 MGF-Net 的同类测试——那次它抓出了三类问题，其中一类还是测试自身写错。

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


def test_identity():
    """pred == target 时 PSNR = inf，SSIM = 1。"""
    x = _phantom()
    assert M.psnr(x, x) == float("inf"), f"PSNR(x,x)={M.psnr(x,x)} 应为 inf"
    assert abs(M.ssim(x, x) - 1.0) < 1e-10, f"SSIM(x,x)={M.ssim(x,x)} 应为 1"
    print("  [1] 一致性：PSNR=inf, SSIM=1                OK")


def test_constant():
    """常量图：相同则 PSNR=inf/SSIM=1；不同则有限且无 NaN。"""
    c = np.full((64, 64), 0.4)
    d = np.full((64, 64), 0.6)
    assert M.psnr(c, c) == float("inf")
    assert abs(M.ssim(c, c) - 1.0) < 1e-10
    p, s = M.psnr(c, d), M.ssim(c, d)
    assert np.isfinite(p) and np.isfinite(s), f"常量图不同值应有限，得 PSNR={p} SSIM={s}"
    print(f"  [2] 常量图无 NaN                          OK  "
          f"(0.4 vs 0.6: PSNR={p:.2f} SSIM={s:.4f})")


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
    """PSNR 的解析校验：均匀误差 ε 时 PSNR = -20·log10(ε)。"""
    x = _phantom(64, 64)
    for eps in [0.01, 0.05, 0.1]:
        y = np.clip(x + eps, 0, 1)
        # 若触及边界则跳过（截断会改变实际误差）
        if np.any((x + eps > 1.0) | (x + eps < 0.0)):
            continue
        expect = -20.0 * np.log10(eps)
        got = M.psnr(y, x)
        assert abs(got - expect) < 1e-6, f"ε={eps}: 得 {got:.6f}，应 {expect:.6f}"
        print(f"  [5] PSNR 解析校验 ε={eps}: {got:.4f} dB (期望 {expect:.4f})  OK")
        return
    print("  [5] PSNR 解析校验  跳过（合成图触边）")


def test_scale_consistency():
    """同一图以 float[0,1] 与 uint8 两种方式喂入，结果应一致。

    这一条直接抓"量纲搞错"这类最隐蔽的错误——
    MGF-Net 项目里就有过 [0,1] 与 [0,255] 混用导致 SD 差 255 倍的实例。
    """
    x = _phantom()
    n = np.clip(x + RNG.normal(0, 0.05, x.shape), 0, 1)
    x8 = np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
    n8 = np.clip(np.rint(n * 255), 0, 255).astype(np.uint8)

    p1, s1 = M.psnr(n, x), M.ssim(n, x)
    p2, s2 = M.psnr(n8, x8), M.ssim(n8, x8)
    # uint8 路径存在量化误差，容差放宽
    assert abs(p1 - p2) < 0.5, f"PSNR 尺度不一致: float={p1:.4f} uint8={p2:.4f}"
    assert abs(s1 - s2) < 5e-3, f"SSIM 尺度不一致: float={s1:.4f} uint8={s2:.4f}"
    print(f"  [6] float[0,1] / uint8 尺度一致性         OK  "
          f"(PSNR {p1:.2f}/{p2:.2f}, SSIM {s1:.4f}/{s2:.4f})")


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
    x = _phantom(256, 256)
    prev = None
    for sig in [0.01, 0.03, 0.05, 0.10]:
        n = np.clip(x + RNG.normal(0, sig, x.shape), 0, 1)
        p, s = M.psnr(n, x), M.ssim(n, x)

        # PSNR 可解析校验：误差近似 N(0, σ²) 时 PSNR ≈ -20·log10(σ)
        # 截断会略微改变实际误差，故放宽到 1 dB
        expect = -20.0 * np.log10(sig)
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
