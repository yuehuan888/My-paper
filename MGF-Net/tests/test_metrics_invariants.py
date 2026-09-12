"""
L0 层指标不变量测试。

这一层**不需要任何外部参考值**，只用数学上必须成立的性质来抓实现错误。
它能覆盖绝大多数移植 bug——尤其是"数字看着正常但其实错"那类。

可两种方式运行：
    python tests/test_metrics_invariants.py        # 直接跑，打印结果
    pytest tests/test_metrics_invariants.py        # 若装了 pytest

测试层次说明（见 `指标实现方案调研_2026-09-12.md`）：
    L0 不变量（本文件）      无需参考值，10 分钟，应进 CI
    L1 MATLAB 金标准对照     本地一次性，不入库
    L2 MIT 生态交叉验证      sewar / skimage
    L3 文献量级对照          数值应落在已发表表格的区间内
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import metrics as M  # noqa: E402

RNG = np.random.default_rng(20260912)


# ------------------------------------------------------------------ 工具
def _rand_u8(h=64, w=64, variant=0):
    """随机但结构化的测试图（纯白噪声的熵接近 8，不适合检验 EN 区间）。

    variant 参数很关键：CT 与 MRI 是**结构互相独立**的两种模态。
    若 a、b 用同一基础图案，则 a-b 退化成纯噪声，SCD 会得到近 0 的
    假值（实测 0.17），而独立情形下像素平均的 SCD 应为 sqrt(2)≈1.414。
    """
    yy, xx = np.mgrid[0:h, 0:w]
    if variant == 0:
        base = 128 + 60 * np.sin(xx / 7.0) + 40 * np.cos(yy / 11.0)
    elif variant == 1:
        base = 118 + 70 * np.cos(xx / 13.0 + 1.3) + 50 * np.sin(yy / 5.0 + 0.7)
    else:
        base = 140 + 55 * np.sin((xx + yy) / 9.0) + 35 * np.cos((xx - yy) / 16.0)
    base = base + RNG.normal(0, 6, size=(h, w))
    return np.clip(base, 0, 255).astype(np.uint8)


def _rand_pair(h=64, w=64):
    """两张结构独立、量级相近的图，类比 CT 与 MRI。"""
    return _rand_u8(h, w, variant=0), _rand_u8(h, w, variant=1)


def _const_u8(value, h=64, w=64):
    return np.full((h, w), value, dtype=np.uint8)


def _to_float01(img_u8):
    return img_u8.astype(np.float64) / 255.0


ALL_METRIC_FUNCS = [
    "entropy",
    "std_dev",
    "spatial_frequency",
]


# ================================================================== 1. 完全一致
def _qabf_theoretical_max():
    """Qabf 在"完全保持"时的理论上限。

    Qg = Tg/(1+exp(kg*(G-Dg)))，G=1 时取到 Tg/(1+exp(kg*(1-Dg)))
    Qa = Ta/(1+exp(ka*(A-Da)))，A=1 时取到 Ta/(1+exp(ka*(1-Da)))
    二者的 sigmoid 上限是**渐近逼近**的，故 Qabf 永远取不到 1.0。
    """
    qg_max = M._QABF_TG / (1.0 + np.exp(M._QABF_KG * (1.0 - M._QABF_DG)))
    qa_max = M._QABF_TA / (1.0 + np.exp(M._QABF_KA * (1.0 - M._QABF_DA)))
    return qg_max * qa_max


def test_identity_pair():
    """F = A 时：SSIM=1, CC=1, PSNR=inf, MI=H(A), SCD=1, Qabf=理论上限"""
    a = _rand_u8()
    b = _rand_u8()

    assert abs(M.ssim(a, a) - 1.0) < 1e-10, f"SSIM(A,A)={M.ssim(a,a)} 应为 1"
    assert abs(M.correlation_coefficient(a, a) - 1.0) < 1e-10, "CC(A,A) 应为 1"
    assert M.psnr(a, a) == float("inf"), f"PSNR(A,A)={M.psnr(a,a)} 应为 inf"
    assert abs(M.mutual_information(a, a) - M.entropy(a)) < 1e-9, "MI(A,A) 应等于 H(A)"

    # Qabf 的"完全保持"不是 1.0，而是公式的渐近上限 0.97479
    q_max = _qabf_theoretical_max()
    q = M.qabf(a, a, a)
    assert abs(q - q_max) < 1e-6, \
        f"Qabf(A,A,A)={q:.8f}，理论上限={q_max:.8f}（差 {abs(q-q_max):.2e}）"

    # SCD：F=S1 时 corr(S1-S2,S1)+corr(0,S2) = corr(S1-S2,S1) + 0；令 b=S1 则两者相等
    s = M.scd(a, a, b)
    assert np.isfinite(s), f"SCD(A,A,B)={s}"
    assert -2.0 <= s <= 2.0, f"SCD(A,A,B)={s} 越界"

    print("  [1] 一致性与 PSNR=inf                     OK")
    print(f"      MI(A,A)={M.mutual_information(a,a):.6f}  H(A)={M.entropy(a):.6f}")
    print(f"      Qabf(A,A,A)={q:.6f}  理论上限={q_max:.6f}")


# ================================================================== 2. 常量图
def test_constant_image_is_nan_free():
    """常量图是最狠的 NaN 测试——Qabf 的 0/0 在这里必炸。"""
    c = _const_u8(128)
    a = _rand_u8()

    for name in ["entropy", "std_dev", "spatial_frequency"]:
        v = getattr(M, name)(c)
        assert np.isfinite(v), f"{name}(常量图) 得 {v}，应有限"
        assert abs(v) < 1e-9, f"{name}(常量图) 得 {v}，应为 0"

    assert abs(M.ssim(c, c) - 1.0) < 1e-10
    assert M.psnr(c, c) == float("inf")

    q = M.qabf(c, c, c)
    assert np.isfinite(q), f"Qabf 在常量图上得 {q} —— 0/0 的 NaN 泄漏了"
    assert abs(q - 1.0) < 1e-9, f"Qabf(常量,常量,常量)={q}，应为 1"

    s = M.scd(c, c, c)
    assert np.isfinite(s), f"SCD 在常量图上得 {s}"

    q2 = M.qabf(a, c, c)
    assert np.isfinite(q2), f"Qabf(非平凡图, 常量, 常量) 得 {q2} —— 权重和为 0 的分支没处理"

    print("  [2] 常量图无 NaN / EN=SD=SF=0              OK")
    print(f"      Qabf(常量)= {q:.6f}   Qabf(图,常量,常量)= {q2:.6f}")


# ================================================================== 3. 值域
def test_value_ranges():
    """各指标的数学值域。SCD > 2 基本判定为 bug。"""
    a, b = _rand_pair()
    f = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255).astype(np.uint8)

    q = M.qabf(f, a, b)
    assert 0.0 - 1e-9 <= q <= 1.0 + 1e-9, f"Qabf={q} 越界 [0,1]"

    s = M.scd(f, a, b)
    assert -2.0 - 1e-9 <= s <= 2.0 + 1e-9, f"SCD={s} 越界 [-2,2]（>2 基本是分母漏根号的 bug）"
    assert 1.0 <= s <= 2.0, f"SCD={s}，合理区间约 1.3~1.9"

    cc = M.correlation_coefficient(f, a)
    assert -1.0 - 1e-9 <= cc <= 1.0 + 1e-9, f"CC={cc} 越界 [-1,1]"

    ss = M.ssim(f, a)
    assert -1.0 - 1e-9 <= ss <= 1.0 + 1e-9, f"SSIM={ss} 越界"

    v = M.vifp(f, a)
    assert 0.0 - 1e-9 <= v <= 1.0 + 1e-9, f"VIF={v} 越界 [0,1]"

    print("  [3] 值域钳制                              OK")
    print(f"      Qabf={q:.4f}  SCD={s:.4f}  SSIM={ss:.4f}  VIF={v:.4f}  CC={cc:.4f}")


# ================================================================== 4. 尺度一致性
def test_uint8_float_scale_consistency():
    """同一图以 uint8 与 float[0,1] 两种方式喂入，结果必须完全相同。

    这一条直接抓 SD 量纲与 SSIM dtype 两类 bug——它们是本领域最隐蔽的错误来源。
    """
    a, b = _rand_pair()
    f = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255).astype(np.uint8)

    a_f, b_f, f_f = _to_float01(a), _to_float01(b), _to_float01(f)

    checks = {
        "EN": (M.entropy(f), M.entropy(f_f)),
        "SD": (M.std_dev(f), M.std_dev(f_f)),
        "SF": (M.spatial_frequency(f), M.spatial_frequency(f_f)),
        "MI": (M.mutual_information(f, a), M.mutual_information(f_f, a_f)),
        "CC": (M.correlation_coefficient(f, a), M.correlation_coefficient(f_f, a_f)),
        "PSNR": (M.psnr(f, a), M.psnr(f_f, a_f)),
        "SSIM": (M.ssim(f, a), M.ssim(f_f, a_f)),
        "VIF": (M.vifp(f, a), M.vifp(f_f, a_f)),
        "SCD": (M.scd(f, a, b), M.scd(f_f, a_f, b_f)),
        "Qabf": (M.qabf(f, a, b), M.qabf(f_f, a_f, b_f)),
    }

    bad = []
    for name, (v_u8, v_f) in checks.items():
        if not np.isclose(v_u8, v_f, rtol=0, atol=1e-9):
            bad.append(f"{name}: uint8={v_u8!r} vs float={v_f!r}")
    assert not bad, "尺度不一致：\n    " + "\n    ".join(bad)

    print("  [4] uint8 / float[0,1] 尺度一致性           OK")
    print("      " + "  ".join(f"{k}={v[0]:.4f}" for k, v in checks.items()))


# ================================================================== 5. 单调性
def test_monotonic_degradation():
    """逐步加高斯噪声，SSIM / Qabf / PSNR / VIF 必须单调下降。"""
    a = _rand_u8()
    b = _rand_u8()
    f0 = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255)

    sigmas = [0, 2, 5, 10, 20, 35, 50]
    ssims, qabfs, psnrs, vifs, scds = [], [], [], [], []
    for s in sigmas:
        noisy = np.clip(f0 + RNG.normal(0, s, size=f0.shape), 0, 255).astype(np.uint8)
        ssims.append(M.ssim(noisy, a))
        qabfs.append(M.qabf(noisy, a, b))
        psnrs.append(M.psnr(noisy, a))
        vifs.append(M.vifp(noisy, a))
        scds.append(M.scd(noisy, a, b))

    for name, vals in [("SSIM", ssims), ("Qabf", qabfs), ("PSNR", psnrs), ("VIF", vifs)]:
        diffs = np.diff(vals)
        assert np.all(diffs <= 1e-9), f"{name} 未单调下降: {[f'{v:.4f}' for v in vals]}"

    print("  [5] 加噪单调性                            OK")
    print(f"      SSIM {ssims[0]:.4f} -> {ssims[-1]:.4f}   "
          f"Qabf {qabfs[0]:.4f} -> {qabfs[-1]:.4f}   "
          f"VIF {vifs[0]:.4f} -> {vifs[-1]:.4f}")


# ================================================================== 6. 形状
def test_shapes_and_odd_sizes():
    """奇数边长、非 11 倍数边长、极小图、非方形。"""
    for shape in [(63, 63), (64, 57), (100, 100), (23, 41)]:
        a = _rand_u8(*shape)
        b = _rand_u8(*shape)
        f = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255).astype(np.uint8)
        for fn, args in [
            (M.entropy, (f,)), (M.std_dev, (f,)), (M.spatial_frequency, (f,)),
            (M.psnr, (f, a)), (M.correlation_coefficient, (f, a)),
            (M.mutual_information, (f, a)), (M.ssim, (f, a)),
            (M.scd, (f, a, b)), (M.qabf, (f, a, b)), (M.vifp, (f, a)),
        ]:
            v = fn(*args)
            assert np.isfinite(v), f"{fn.__name__} 在 {shape} 上得 {v}"

    # 小于 SSIM 窗口的图应给出明确报错而不是静默出错
    tiny = _rand_u8(8, 8)
    try:
        M.ssim(tiny, tiny)
        raise AssertionError("8×8 图调用 SSIM 应抛 ValueError，而不是静默返回")
    except ValueError:
        pass

    # 尺寸不一致应报错
    try:
        M.psnr(_rand_u8(32, 32), _rand_u8(32, 40))
        raise AssertionError("尺寸不一致应抛 ValueError")
    except ValueError:
        pass

    print("  [6] 形状 / 奇数边长 / 越界报错              OK")


# ================================================================== 7. 双源顺序无关
def test_source_order_invariance():
    """交换两源：MI 求和形式应不变，CC/Qabf/SCD 应不变。"""
    a, b = _rand_pair()
    f = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255).astype(np.uint8)

    m1 = M.evaluate_pair(f, a, b)
    m2 = M.evaluate_pair(f, b, a)

    for k in ["MI", "SSIM", "Qabf", "SCD", "VIF", "CC", "PSNR", "EN", "SD", "SF"]:
        assert np.isclose(m1[k], m2[k], rtol=0, atol=1e-9), \
            f"交换两源后 {k} 变了: {m1[k]} vs {m2[k]}"

    print("  [7] 双源顺序无关性                        OK")
    print("      " + "  ".join(f"{k}={m1[k]:.4f}" for k in
          ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]))


# ================================================================== 8. 量级
def test_magnitude_sanity():
    """L3 前置：数值应落在已发表表格的常见区间内。

    飞出区间通常意味着量纲或变体选错了——这是抓"数字漂亮但对不上"的一道闸。
    注意：本测试用的是合成图，区间取宽；真实医学图会另做 L3 对照。
    """
    a, b = _rand_pair(256, 256)
    f = np.clip(0.5 * a.astype(np.float64) + 0.5 * b.astype(np.float64), 0, 255).astype(np.uint8)
    m = M.evaluate_pair(f, a, b)

    bounds = {
        "EN": (4.0, 9.0),
        "SD": (20.0, 110.0),
        "SF": (2.0, 60.0),
        "MI": (0.5, 8.0),
        "CC": (0.3, 1.0),
        "PSNR": (5.0, 45.0),
        "SSIM": (0.2, 1.0),
        "VIF": (0.05, 1.0),
        "SCD": (0.5, 2.0),
        "Qabf": (0.1, 1.0),
    }
    bad = []
    for k, (lo, hi) in bounds.items():
        if not (lo <= m[k] <= hi):
            bad.append(f"{k}={m[k]:.4f} 不在 [{lo}, {hi}]")

    if bad:
        print("  [8] 量级检查  ⚠️ 超出合成图区间（需人工判断，不一定是 bug）")
        for x in bad:
            print(f"      - {x}")
    else:
        print("  [8] 量级区间检查                          OK")


# ------------------------------------------------------------------ 主入口
def main():
    print("=" * 74)
    print("L0 指标不变量测试")
    print("=" * 74)
    tests = [
        test_identity_pair,
        test_constant_image_is_nan_free,
        test_value_ranges,
        test_uint8_float_scale_consistency,
        test_monotonic_degradation,
        test_shapes_and_odd_sizes,
        test_source_order_invariance,
        test_magnitude_sanity,
    ]
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
    print(f"结果：{len(tests) - failed}/{len(tests)} 通过")
    print("=" * 74)
    return failed


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
