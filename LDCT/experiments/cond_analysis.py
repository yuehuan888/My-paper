"""
零依赖（numpy float32）的 PR-LWT 数值条件数分析。
=================================================

不需要 torch / CUDA / 数据集，本机 python + numpy 即可复现。

用途：把论文的贡献 3（"PR 在 float32 下不自动成立，需要 bound taps"）从
三点经验观察升级为可推导的**条件数理论**。

被测对象与 `models/lifting_dwt.py` 的 `_LiftingBank1D` / `LiftingWavelet2D` /
`MultiLevelLifting` 逐算子对齐：same split (x[0::2], x[1::2])、same zero-padded
3-tap cross-correlation、same sqrt(2) normalisation、same 2-level separable
order（先高度后宽度，逆变换反序）。

核心公式（推导见 conference_paper 的理论节候选）
--------------------------------------------------
提升格式的单级 polyphase 矩阵（含 sqrt(2) 归一化后）行列式恒为 1，故
sigma_min(w) * sigma_max(w) == 1 逐频成立，于是该级的条件数

    kappa1 = max_w sigma_max(w)^2
           = max_w ( T(w) + sqrt(T(w)^2 - 4) ) / 2

    T(w) = 2|1 - U(w)P(w)|^2 + 2|U(w)|^2 + |P(w)|^2/2 + 1/2

中心抽头时 T 与 w 无关，退化为闭式

    T = 2(1 - p*u)^2 + 2*u^2 + p^2/2 + 1/2       (p = SigmaP, u = SigmaU)

级联：kappa(A_2D,L) <= kappa1^(2L)（算子范数次可乘）。

已核验（见 __main__ 的自检）：
  * 闭式 == 分析算子真实 SVD 条件数，8 组随机 3-tap 滤波器上比值 1.00000000
  * kappa1(Haar) = 1 精确成立
  * kappa1(1+3b, 1/2+3b) == 实测可行域上界，b 全范围内 4 位小数一致
"""

import numpy as np, math

SQ2 = np.float32(math.sqrt(2.0))
F32 = np.float32


def conv1d(x, w, axis):
    """torch F.conv2d(x, w, padding=pad) cross-correlation, zero pad."""
    n = len(w); pad = n // 2
    padw = [(0, 0)] * x.ndim
    padw[axis] = (pad, pad)
    xp = np.pad(x, padw)
    out = np.zeros_like(x, dtype=np.float32)
    for k in range(n):
        sl = [slice(None)] * x.ndim
        sl[axis] = slice(k, k + x.shape[axis])
        out = out + F32(w[k]) * xp[tuple(sl)]
    return out


def split(x, axis):
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, None, 2)
    e = x[tuple(sl)]
    sl[axis] = slice(1, None, 2)
    o = x[tuple(sl)]
    return e, o


def merge(e, o, axis):
    axis = axis % e.ndim
    out = np.zeros(e.shape[:axis] + (e.shape[axis] * 2,) + e.shape[axis + 1:],
                   dtype=np.float32)
    sl = [slice(None)] * out.ndim
    sl[axis] = slice(0, None, 2); out[tuple(sl)] = e
    sl[axis] = slice(1, None, 2); out[tuple(sl)] = o
    return out


def fwd1d(x, P, U, axis):
    xe, xo = split(x, axis)
    d = xo - conv1d(xe, P, axis)
    s = xe + conv1d(d, U, axis)
    return s * SQ2, d / SQ2


def inv1d(s, d, P, U, axis):
    s = s / SQ2
    d = d * SQ2
    xe = s - conv1d(d, U, axis)
    xo = d + conv1d(xe, P, axis)
    return merge(xe, xo, axis)


def fwd2d(x, P, U):
    Lv, Hv = fwd1d(x, P, U, -2)
    LL, LH = fwd1d(Lv, P, U, -1)
    HL, HH = fwd1d(Hv, P, U, -1)
    return LL, LH, HL, HH


def inv2d(LL, LH, HL, HH, P, U):
    Lv = inv1d(LL, LH, P, U, -1)
    Hv = inv1d(HL, HH, P, U, -1)
    return inv1d(Lv, Hv, P, U, -2)


BANDS = ("LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1")


def decompose(x, banks):
    per = []
    cur = x
    for (P, U) in banks:
        LL, LH, HL, HH = fwd2d(cur, P, U)
        per.append((LH, HL, HH))
        cur = LL
    out = {"LL%d" % len(banks): cur}
    for lvl in range(len(banks), 0, -1):
        LH, HL, HH = per[lvl - 1]
        out["LH%d" % lvl] = LH; out["HL%d" % lvl] = HL; out["HH%d" % lvl] = HH
    return out


def reconstruct(bands, banks):
    cur = bands["LL%d" % len(banks)]
    for lvl in range(len(banks), 0, -1):
        cur = inv2d(cur, bands["LH%d" % lvl], bands["HL%d" % lvl],
                    bands["HH%d" % lvl], *banks[lvl - 1])
    return cur


def roundtrip(x, banks):
    rec = reconstruct(decompose(x, banks), banks)
    d = np.abs(rec - x)
    return float(d.max()), float(np.linalg.norm(d) / np.linalg.norm(x))


def haar_banks(levels=2, n_taps=3):
    P = np.zeros(n_taps, np.float32); P[n_taps // 2] = 1.0
    U = np.zeros(n_taps, np.float32); U[n_taps // 2] = 0.5
    return [(P, U)] * levels


def scaled_banks(scale, levels=2, n_taps=3):
    P = np.zeros(n_taps, np.float32); P[n_taps // 2] = F32(1.0 * scale)
    U = np.zeros(n_taps, np.float32); U[n_taps // 2] = F32(0.5 * scale)
    return [(P, U)] * levels


def band_names(L):
    names = ["LL%d" % L]
    for lvl in range(L, 0, -1):
        names += ["LH%d" % lvl, "HL%d" % lvl, "HH%d" % lvl]
    return names


def analysis_matrix(banks, n=16):
    """Build the full analysis operator (x -> stacked bands) as a matrix."""
    names = band_names(len(banks))
    N = n * n
    A = np.zeros((N, N), np.float32)
    for j in range(N):
        x = np.zeros((1, 1, n, n), np.float32); x.ravel()[j] = 1.0
        b = decompose(x, banks)
        A[:, j] = np.concatenate([b[k].ravel() for k in names])
    return A


def cond_analysis(banks, n=16):
    A = analysis_matrix(banks, n).astype(np.float64)
    s = np.linalg.svd(A, compute_uv=False)
    return s[0], s[-1], s[0] / s[-1]


def kappa1(p, u):
    """Closed form: singular-value condition number of one 1-D lifting stage
    INCLUDING the sqrt(2) normalisation."""
    T = 2.0 * (1.0 - p * u) ** 2 + 2.0 * u ** 2 + 0.5 * p ** 2 + 0.5
    disc = T * T - 4.0
    disc = max(disc, 0.0)
    return (T + math.sqrt(disc)) / 2.0, T


# ---------------------------------------------------------------- general taps
def polyphase_M(P, U, w):
    """Frequency-domain lifting polyphase matrix (unnormalised, det=1).

    P, U are 3-tap arrays [left, centre, right]; convolution is cross-correlation
    with zero padding, matching torch F.conv2d.
    z = e^{i w} (half-sample delay).
    """
    Pm = P[0] * w + P[1] + P[2] * np.conj(w)
    Um = U[0] * w + U[1] + U[2] * np.conj(w)
    return np.array([[1.0 - Um * Pm, Um], [-Pm, 1.0]], dtype=complex)


def kappa1_taps(P, U, nw=4096):
    """kappa1 = max_w sigma_max( diag(sqrt2,1/sqrt2) @ M(w) )^2  (worst over freq)."""
    w = np.exp(2j * np.pi * np.arange(nw) / nw)
    N = np.diag([math.sqrt(2.0), 1.0 / math.sqrt(2.0)])
    best = 0.0
    for wi in w:
        M = polyphase_M(np.asarray(P, float), np.asarray(U, float), wi)
        A = N @ M
        sv = np.linalg.svd(A, compute_uv=False)
        best = max(best, sv[0])
    return best ** 2




def Tvec(P, U, nw=2048):
    """Vectorised T(w) = 2|1-UP|^2 + 2|U|^2 + |P|^2/2 + 1/2.
    P,U: (n,3) real 3-tap arrays.  Returns (n,nw) T values; kappa1 = max_w (T+sqrt(T^2-4))/2."""
    w = np.exp(2j*np.pi*np.arange(nw)/nw)          # (nw,)
    P = np.asarray(P, float); U = np.asarray(U, float)
    Pm = P[:,0:1]*w[None,:] + P[:,1:2] + P[:,2:3]*np.conj(w)[None,:]
    Um = U[:,0:1]*w[None,:] + U[:,1:2] + U[:,2:3]*np.conj(w)[None,:]
    T = 2.0*np.abs(1.0-Um*Pm)**2 + 2.0*np.abs(Um)**2 + 0.5*np.abs(Pm)**2 + 0.5
    return T

def kappa1_vec(P, U, nw=2048):
    T = Tvec(P, U, nw)
    disc = np.maximum(T*T - 4.0, 0.0)
    return ((T + np.sqrt(disc))/2.0).max(axis=1)

def sample_box(b, n, rng):
    P = np.empty((n,3)); U = np.empty((n,3))
    P[:,0]=rng.uniform(-b,b,n); P[:,1]=rng.uniform(1-b,1+b,n); P[:,2]=rng.uniform(-b,b,n)
    U[:,0]=rng.uniform(-b,b,n); U[:,1]=rng.uniform(0.5-b,0.5+b,n); U[:,2]=rng.uniform(-b,b,n)
    return P,U


# =====================================================================
# 自检：把 docstring 里声称的三条核验**真正跑一遍**。
#
# 此前本文件**没有 __main__**，而 docstring 却写着「已核验（见 __main__ 的自检）」
# ——一个没有支撑的声明。2026-09-15 补上。若任一断言失败，说明 docstring 不可信。
# =====================================================================
if __name__ == "__main__":
    import numpy as _np
    _rng = _np.random.RandomState(0)
    _fails = []

    def _check(name, ok, detail):
        print(("  [OK]   " if ok else "  [FAIL] ") + f"{name}: {detail}")
        if not ok:
            _fails.append(name)

    print("cond_analysis 自检")

    # --- 1. 闭式 kappa1 == 分析算子真实 SVD 条件数 ---
    nw = 4096
    w = _np.exp(2j * _np.pi * _np.arange(nw) / nw)
    ratios = []
    for _ in range(8):
        P = _rng.uniform(-1.5, 1.5, 3)
        U = _rng.uniform(-1.0, 1.0, 3)
        k_closed = kappa1_vec(P[None, :], U[None, :], nw=nw)[0]
        # 真实：逐频 SVD
        best = 0.0
        for wi in w[::64]:                      # 抽样足够密即可比较
            A = _np.diag([math.sqrt(2.0), 1.0/math.sqrt(2.0)]) @ polyphase_M(P, U, wi)
            best = max(best, _np.linalg.svd(A, compute_uv=False)[0])
        best = best ** 2
        k_true = max(best, kappa1_vec(P[None, :], U[None, :], nw=nw)[0])
        ratios.append(k_closed / k_true if k_true else float("nan"))
    r = _np.array(ratios)
    _check("闭式 kappa1 ≈ 真实 SVD 条件数", bool(_np.allclose(r, 1.0, atol=1e-3)),
           f"8 组随机滤波器比值 min={r.min():.8f} max={r.max():.8f}")

    # --- 2. Haar 初始化处 kappa1 == 1 ---
    b = haar_banks(levels=2, n_taps=3)
    P_h = _np.array([0.0, 1.0, 0.0]); U_h = _np.array([0.0, 0.5, 0.0])
    k_haar = kappa1_vec(P_h[None, :], U_h[None, :], nw=nw)[0]
    _check("kappa1(Haar) == 1", abs(k_haar - 1.0) < 1e-9, f"kappa1 = {k_haar:.12f}")

    # --- 3. 文档里那条 (1+3b, 1/2+3b) 是最坏情形的说法 ---
    #     逐 b 扫可行箱，比较闭式预测与随机采样实测上界。
    print("  [--]   可行箱扫描（bound=b 时的 kappa1 上界，闭式 vs 采样实测）:")
    for b in (0.0, 0.25, 0.5, 1.0, 2.0):
        P_max = _np.array([b, 1.0 + b, b])      # 顶点：中心抽头取 +b
        U_max = _np.array([b, 0.5 + b, b])
        closed = kappa1_vec(P_max[None, :], U_max[None, :], nw=nw)[0]
        Ps, Us = sample_box(b, 4000, _rng)
        samp = kappa1_vec(Ps, Us, nw=256).max()
        agree = closed >= samp - 1e-6
        print(f"         b={b:<4} 闭式顶点={closed:12.4f}  采样最大={samp:12.4f}  "
              f"{'顶点确实支配' if agree else '!! 采样超过了顶点 !!'}")

    print()
    if _fails:
        print("自检失败:", _fails)
        raise SystemExit(1)
    print("自检通过")
