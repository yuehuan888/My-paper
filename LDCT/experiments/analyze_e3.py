"""E3 分析：把 taps 初始化在离 Haar 距离 d 处，看训练后落到哪。

要回答的问题
================================================================
论文的负结果是"无约束可学习小波 ≈ 固定 Haar"，机制解释是"学习摧毁了可逆性"。
但还有一层更根本的可能：**这个任务的最优小波本来就在 Haar 附近**，
所以梯度没有动力把 taps 推远。

E3 从反面检验：主动把 taps 初始到远处（d = 0 / 0.5 / 1 / 2 / 4），看训练把它带到哪。
  - 若不同起点都收敛到**相近且很小的 drift** -> 支持"Haar 附近就是最优"
  - 若收敛到同一个**非零**的 drift      -> 存在**偏好工作点**（比 Haar 略偏）
  - 若各处基本不动                       -> 起点有残留影响，训练不足以重塑变换

注意：d=2 起闭环已退化到 1.3e-02，d=4 更是 3.8e+01（float32 下已崩）。
所以 d 大的两档还顺带在测**训练能否从数值已崩的状态恢复**。
"""

from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models.denoiser import PRWaveletDenoiser  # noqa: E402

BOUND = 8.0
LEVELS = [(0.0, "d00"), (0.5, "d05"), (1.0, "d10"), (2.0, "d20"), (4.0, "d40")]


def initial_roundtrip(d: float) -> float:
    """构造后立刻测闭环（未训练），用于说明该起点的数值状态。"""
    m = PRWaveletDenoiser(bound=BOUND, init_drift=d)
    g = torch.Generator().manual_seed(7)
    x = torch.rand(1, 1, 256, 256, generator=g)
    return m.transform_roundtrip(x)[1]


def main():
    print("=" * 76)
    print("E3: taps 初始化距离 -> 训练后距离  (bound=8.0, 允许 |taps| <= 9)")
    print("=" * 76)
    print("  %-6s %-13s %-13s %-11s %-11s %s"
          % ("init", "初始闭环", "训练后drift", "变化", "PSNR", "判读"))
    rows = []
    for d, tag in LEVELS:
        dr, ps = [], []
        for s in range(3):
            ck = os.path.join(HERE, "runs", f"e3_{tag}_s{s}", "best.pth")
            if not os.path.exists(ck):
                continue
            st = torch.load(ck, map_location="cpu", weights_only=False)
            m = PRWaveletDenoiser(bound=BOUND, init_drift=d)
            m.load_state_dict(st.get("model") or st.get("model_state_dict"))
            dr.append(m.wavelet.max_drift())
            rp = os.path.join(HERE, "runs", f"e3_{tag}_s{s}", "results.json")
            if os.path.exists(rp):
                with io.open(rp, encoding="utf-8") as f:
                    ps.append(json.load(f)["test_mean"]["PSNR"])
        if not dr:
            print("  %-6s (未完成)" % d)
            continue
        md = float(np.mean(dr))
        mp = float(np.mean(ps)) if ps else float("nan")
        ch = md - d
        if abs(ch) < 0.15:
            note = "基本不动"
        elif ch > 0:
            note = "训练把它推远"
        else:
            note = "训练把它收回"
        rows.append((d, md, ch, mp))
        print("  %-6s %-13.2e %-13.4f %-11s %-11.4f %s"
              % (d, initial_roundtrip(d), md, "%+.4f" % ch, mp, note))

    if len(rows) < 2:
        print("\n(数据不足，等批次跑完)")
        return 0

    # ------------------------------------------------------------------
    # ⚠️ 判读必须**分段**：d=4 的起点前向已经数值崩溃（闭环 rel_L2 >> 1），
    #    它的"训练后 drift 不动"与 d<=2 的"被拉向工作点"是**两种不同现象**。
    #    混在一起算范围/相关会被 d=4 完全带偏（实测：一并统计得 r=+0.871、
    #    结论误判为"起点有残留影响"；分段后 d<=2 的真实相关是负的）。
    # ------------------------------------------------------------------
    def is_broken(d):
        """起点是否已数值崩溃：闭环 rel_L2 > 1 意味着重构误差比信号还大。"""
        return initial_roundtrip(d) > 1.0

    healthy = [r for r in rows if not is_broken(r[0])]
    broken = [r for r in rows if is_broken(r[0])]

    if len(healthy) >= 2:
        ds = np.array([r[0] for r in healthy])
        md = np.array([r[1] for r in healthy])
        print()
        print("  【可恢复区间】起点闭环 << 1 的 %d 档:" % len(healthy))
        print("    训练后 drift 范围: %.4f - %.4f  (跨度 %.4f)"
              % (md.min(), md.max(), md.max() - md.min()))
        print("    起点与终点相关: r = %+.3f" % np.corrcoef(ds, md)[0, 1])
        print()
        if md.max() - md.min() < 0.5:
            print("    ✓ 不同起点收敛到**相近的 drift** -> 存在偏好工作点，")
            print("      起点的影响基本被训练抹掉（相关为负 = 高起点被压低、")
            print("      低起点被抬高，正是向共同点收敛的特征）。")
            print("      该点均值 %.3f：" % md.mean())
            if md.mean() < 0.3:
                print("        接近 0 -> 支持「Haar 附近即最优」。")
            else:
                print("        **明显非零** -> 最优小波**略偏离 Haar**，")
                print("        不能说「最优就是 Haar」。")
        else:
            print("    ✗ 跨度较大 -> 起点有残留影响。")

    if broken:
        print()
        print("  【数值崩溃区间】起点闭环 >> 1 的 %d 档:" % len(broken))
        for d, m_, _c, p in broken:
            stuck = abs(m_ - d) < 0.1
            print("    d=%.1f: 起点闭环 %.1e，训练后 drift=%.4f，PSNR=%.4f"
                  % (d, initial_roundtrip(d), m_, p))
            if stuck:
                print("        -> **训练完全无法自救**（drift 一动不动，PSNR 落到 %.1f dB，"
                      % p)
                print("           即输出全是垃圾）。前向已崩 -> 系数是噪声 -> 梯度无意义。")
                print("           这不是斜坡而是**悬崖**：一旦跨过，训练无法回头。")
            else:
                print("        -> 已部分恢复。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
