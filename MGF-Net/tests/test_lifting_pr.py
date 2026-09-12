"""
PR-LWT 的完美重构验证。

按《发表路线与实验计划_修订版》§3.2 的验收要求：
  - 输入类型：随机图、常量图、脉冲图、奇偶尺寸
  - 参数状态：初始值、大幅扰动、训练后、极端值
  - 报告：最大绝对误差 与 相对 L2 误差（不只报 PSNR）
  - AMP 误差单列

核心命题：**无论 P、U 取什么值，inverse(forward(x)) == x**。

⚠️ 但"可逆"不等于"数值稳定"。本测试额外验证：
    - 参数有界化后，float32 下的闭环误差是否始终可控
    - 若无界，误差如何随参数幅度恶化（对照，证明有界化的必要性）

运行：
    python tests/test_lifting_pr.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lifting_dwt import LiftingWavelet2D, MultiLevelLifting  # noqa: E402

SEP = "=" * 78
SQRT2 = math.sqrt(2.0)


def make_inputs(h: int = 256, w: int = 256, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    return {
        "随机图": torch.rand(1, 1, h, w, generator=g),
        "常量图 0.5": torch.full((1, 1, h, w), 0.5),
        "脉冲图": torch.zeros(1, 1, h, w).index_put_(
            (torch.tensor([0]), torch.tensor([0]),
             torch.tensor([h // 2]), torch.tensor([w // 2])), torch.tensor(1.0)),
        "低频渐变": torch.linspace(0, 1, w).view(1, 1, 1, w).expand(1, 1, h, w).contiguous(),
        "棋盘高频": ((torch.arange(h).view(-1, 1) + torch.arange(w).view(1, -1)) % 2)
                    .float().view(1, 1, h, w),
    }


def roundtrip(mod, x):
    bands = mod.decompose(x)
    rec = mod.reconstruct(bands)
    d = (rec - x).abs()
    rel = (torch.norm(d) / torch.norm(x)).item() if torch.norm(x) > 0 else 0.0
    return d.max().item(), rel


def table(title, mod, inputs, thresh=1e-4):
    print(f"\n  {title}")
    print(f"    {'输入':<12}{'max|err|':>14}{'rel_L2':>14}   判定")
    print("    " + "-" * 58)
    worst = 0.0
    for name, x in inputs.items():
        with torch.no_grad():
            me, rl = roundtrip(mod, x)
        worst = max(worst, me)
        print(f"    {name:<12}{me:>14.3e}{rl:>14.3e}   "
              f"{'✅' if me < thresh else '🔴'}")
    return worst


def all_taps(m):
    out = []
    for bank in m.banks:
        for b in (bank.vert, bank.horiz):
            out += [b.P, b.U]
    return out


def main():
    torch.manual_seed(20260912)
    print(SEP)
    print("PR-LWT 完美重构验证")
    print(SEP)
    inputs = make_inputs()

    # ---------------------------------------------------------- 1 初始
    print("\n【1】初始参数（应为正交归一 Haar 的等价提升形式）")
    m0 = MultiLevelLifting(2, 3, True).eval()
    print(f"    可学习参数量: {m0.param_count()}  （预期 24 = 2级 × 2方向 × 2滤波器 × 3taps）")
    print(f"    初始 taps 偏离 Haar: {m0.max_drift():.3e}  （应为 0，θ 初始化为 0）")
    w1 = table("初始参数下的闭环", m0, inputs)

    # ---------------------------------------------------------- 2 大幅扰动 θ
    print("\n【2】θ 被**大幅扰动**后（θ 是实际被优化的量）")
    m2 = MultiLevelLifting(2, 3, True).eval()
    with torch.no_grad():
        for p in m2.parameters():
            p.add_(torch.randn_like(p) * 50.0)      # 极端大的 θ
    taps = torch.cat([t.flatten() for t in all_taps(m2)])
    print(f"    扰动后 |taps| 范围: [{taps.min():.4f}, {taps.max():.4f}]"
          f"  （bound=0.5，应落在初始值 ±0.5 内）")
    print(f"    taps 相对 Haar 的最大偏离: {m2.max_drift():.4f}（上限 0.5）")
    w2 = table("大幅扰动 θ 后的闭环", m2, inputs)

    # ---------------------------------------------------------- 3 训练后
    print("\n【3】经梯度训练后")
    m3 = MultiLevelLifting(2, 3, True)
    m3.train()
    opt = torch.optim.Adam(m3.parameters(), lr=0.05)
    x = torch.rand(2, 1, 64, 64)
    for _ in range(300):
        opt.zero_grad()
        b = m3.decompose(x)
        # 破坏性目标：把高频子带压到接近 0，逼参数远离初始值
        loss = b["HH1"].pow(2).mean() + b["HL1"].pow(2).mean() + b["LH1"].pow(2).mean()
        loss.backward()
        opt.step()
    print(f"    训练后 taps 相对 Haar 的最大偏离: {m3.max_drift():.4f}"
          f"（{'> 0 说明参数确实被优化了' if m3.max_drift() > 1e-6 else '⚠️ 参数没动'}）")
    m3.eval()
    w3 = table("训练后的闭环", m3, inputs)

    # ---------------------------------------------------------- 4 奇偶尺寸
    print("\n【4】奇数与非 2 的幂尺寸")
    m4 = MultiLevelLifting(2, 3, True).eval()
    print(f"    {'尺寸':<12}{'max|err|':>14}{'rel_L2':>14}   判定")
    print("    " + "-" * 58)
    for shape in [(255, 255), (63, 63), (100, 100), (127, 253), (32, 32), (16, 16)]:
        h, w = shape
        x = torch.rand(1, 1, h, w)
        ph, pw = (4 - h % 4) % 4, (4 - w % 4) % 4
        xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        with torch.no_grad():
            b = m4.decompose(xp)
            rec = m4.reconstruct(b)[:, :, :h, :w]
        d = (rec - x).abs()
        me = d.max().item()
        rl = (torch.norm(d) / torch.norm(x)).item()
        print(f"    {str(shape):<12}{me:>14.3e}{rl:>14.3e}   {'✅' if me < 1e-4 else '🔴'}")

    # ---------------------------------------------------------- 5 Haar 等价性
    print("\n【5】初始化是否等价于正交归一 Haar")
    lv = LiftingWavelet2D(3, learnable=False).eval()
    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        LL, LH, HL, HH = lv(x)

    # 参照：直接按定义算正交归一 Haar
    #   沿高度(dim0)：lo = (a+b)/√2, hi = (b-a)/√2
    def haar1d(a, axis):
        if axis == -2:
            e, o = a[..., 0::2, :], a[..., 1::2, :]
        else:
            e, o = a[..., 0::2], a[..., 1::2]
        return (e + o) / SQRT2, (o - e) / SQRT2

    hL, hH = haar1d(x, -2)         # 高度：低通 / 高通
    eLL, eLH = haar1d(hL, -1)      # 高度低通 再沿宽度 → LL / LH
    eHL, eHH = haar1d(hH, -1)      # 高度高通 再沿宽度 → HL / HH

    for name, got, exp in [("LL", LL, eLL), ("LH", LH, eLH),
                           ("HL", HL, eHL), ("HH", HH, eHH)]:
        print(f"    |{name} - 正交Haar_{name}| 最大 = {(got - exp).abs().max().item():.3e}")
    errs = [max((g - e).abs().max().item() for g, e in
                [(LL, eLL), (LH, eLH), (HL, eHL), (HH, eHH)])]
    print(f"    判定: {'✅ 初始化严格等价于正交归一 Haar' if errs[0] < 1e-6 else '🔴 不匹配'}")

    # ---------------------------------------------------------- 6 有界化的必要性
    print("\n【6】对照：若无参数有界化，float32 闭环如何随参数幅度恶化")
    print("    （本实现已强制有界，此处直接改 θ 的等效 taps 以模拟无界情形）")
    print(f"    {'|taps|max':>12}{'max|err|':>14}{'rel_L2':>14}")
    print("    " + "-" * 42)
    for scale in [1.0, 2.0, 3.5, 5.3]:
        m = MultiLevelLifting(2, 3, True).eval()
        with torch.no_grad():
            for bank in m.banks:
                for b in (bank.vert, bank.horiz):
                    b.p_init.mul_(scale)
                    b.u_init.mul_(scale)
                    b.bound = 0.0            # 关掉有界，等价于无约束
                    b.theta_P.zero_(); b.theta_U.zero_()
        with torch.no_grad():
            me, rl = roundtrip(m, inputs["随机图"])
        tmax = max(t.abs().max().item() for t in all_taps(m))
        print(f"    {tmax:>12.2f}{me:>14.3e}{rl:>14.3e}")

    # ---------------------------------------------------------- 7 极端 θ
    print("\n【7】θ 取极端值（±1e3）时闭环 —— 检验有界化是否守住数值稳定")
    for theta in [1e3, -1e3]:
        m = MultiLevelLifting(2, 3, True).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(theta)
        with torch.no_grad():
            me, rl = roundtrip(m, inputs["随机图"])
        tmax = max(t.abs().max().item() for t in all_taps(m))
        print(f"    θ={theta:+.0e}  |taps|max={tmax:.3f}  "
              f"max|err|={me:.3e}  rel_L2={rl:.3e}  {'✅' if me < 1e-4 else '🔴'}")

    # ---------------------------------------------------------- 8 AMP
    print("\n【8】混合精度（AMP）下的闭环误差 —— 单列，不与 FP32 混同")
    if torch.cuda.is_available():
        m8 = MultiLevelLifting(2, 3, True).cuda().eval()
        for name in ["随机图", "常量图 0.5", "脉冲图"]:
            xc = inputs[name].cuda()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                b = m8.decompose(xc)
                rec = m8.reconstruct(b)
            d = (rec.float() - xc.float()).abs()
            print(f"    {name:<12} max|err| = {d.max().item():.3e}   "
                  f"rel_L2 = {(torch.norm(d)/torch.norm(xc)).item():.3e}")
        print("    注：AMP 引入额外舍入，误差高于 FP32 属正常；")
        print("        若达 1e-2 量级则说明数值稳定性有问题。")
    else:
        print("    （本机无 CUDA，跳过）")

    # ---------------------------------------------------------- 结论
    print("\n" + SEP)
    worst = max(w1, w2, w3)
    print(f"结论：初始 / 大幅扰动 / 训练后 三种状态下最差 max|err| = {worst:.3e}")
    if worst < 1e-4:
        print("✅ PR 在 float32 下始终成立，且参数有界化守住了数值稳定。")
    else:
        print("🔴 存在 >1e-4 的闭环误差，需排查。")


if __name__ == "__main__":
    main()
