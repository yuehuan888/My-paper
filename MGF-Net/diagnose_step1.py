"""
Step-1 审计脚本 —— 在改动任何代码之前，先量化现有实现的问题。

回答四个问题：
  Q1  训练用 learnable_dwt=True、推理用 False，checkpoint 里的可学习滤波器
      究竟有没有被正确载入？（决定现有全部结果是否有效）
  Q2  现有 DWT/IDWT 的单元闭环误差有多大？（PR 性质是否真的被破坏）
  Q3  门控融合的 CT 偏置有多严重？（能否取到等权平均、能否取到纯 MRI）
  Q4  test.py 的全图统计 SSIM 与标准局部窗口 SSIM 差多少？
      Average 在两种口径下分别是什么水平？

运行： D:/DeveloperTools/miniconda/envs/mgfnet/python.exe diagnose_step1.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.mgf_net import MGFNet
from PIL import Image

SEP = "=" * 72


def load_img(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


# ---------------------------------------------------------------- SSIM 实现
def ssim_global(img1, img2, L=1.0):
    """test.py 当前的实现：全图均值/方差 —— 等价于窗口=整张图的退化 SSIM。"""
    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu1, mu2 = np.mean(img1), np.mean(img2)
    s1, s2 = np.var(img1), np.var(img2)
    s12 = np.mean((img1 - mu1) * (img2 - mu2))
    return ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / (
        (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    )


def ssim_local(img1, img2, L=1.0, win=11, sigma=1.5):
    """标准局部窗口 SSIM（高斯窗），领域通用做法。"""
    from scipy.ndimage import gaussian_filter

    C1, C2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    mu1 = gaussian_filter(img1, sigma, truncate=(win // 2) / sigma)
    mu2 = gaussian_filter(img2, sigma, truncate=(win // 2) / sigma)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    s1 = gaussian_filter(img1**2, sigma) - mu1_sq
    s2 = gaussian_filter(img2**2, sigma) - mu2_sq
    s12 = gaussian_filter(img1 * img2, sigma) - mu1_mu2
    m = ((2 * mu1_mu2 + C1) * (2 * s12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
    )
    return float(m.mean())


# ---------------------------------------------------------------- Q1
def q1_checkpoint_filters():
    print(SEP)
    print("Q1  可学习滤波器在推理时是否被正确载入")
    print(SEP)

    ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]

    m_learn = MGFNet(in_channels=1, mid_channels=config.mid_channels, learnable_dwt=True)
    m_fixed = MGFNet(in_channels=1, mid_channels=config.mid_channels, learnable_dwt=False)

    kl = set(m_learn.state_dict().keys())
    kf = set(m_fixed.state_dict().keys())
    print(f"  checkpoint 里 epoch={ckpt.get('epoch')}  loss={ckpt.get('loss')}")
    print(f"  learnable 模型 key 数 : {len(kl)}")
    print(f"  fixed     模型 key 数 : {len(kf)}")
    print(f"  learnable 有而 fixed 没有: {sorted(kl - kf) or '无'}")
    print(f"  fixed 有而 learnable 没有: {sorted(kf - kl) or '无'}")

    # 载入到 fixed 模型（复刻 test.py 的路径）
    try:
        m_fixed.load_state_dict(sd)
        loaded_ok = True
        err = None
    except Exception as e:
        loaded_ok = False
        err = e
    print(f"\n  test.py 路径 load_state_dict(strict=True) -> {'成功' if loaded_ok else '失败'}")
    if not loaded_ok:
        print(f"    异常: {err}")

    # 把 checkpoint 的滤波器值读到 learnable 模型，作为"真值"
    m_learn.load_state_dict(sd)
    tgt = m_learn.dwt.dwt1.filters.detach()

    # fixed 模型载入后的滤波器值
    got = m_fixed.dwt.dwt1.filters.detach()

    haar = torch.tensor(
        [[0.5, 0.5], [0.5, 0.5]], dtype=torch.float32
    )  # 占位，下面重算
    from models.dwt_layer import _haar_filters

    haar = _haar_filters()

    print(f"\n  载入后 fixed 模型 dwt1.filters[0] (LL):\n{got[0, 0]}")
    print(f"  checkpoint 中同位置的值      (LL):\n{tgt[0, 0]}")
    print(f"  纯 Haar 初始值              (LL):\n{haar[0, 0]}")
    print(f"\n  |载入值 - checkpoint 值| 最大 = {(got - tgt).abs().max():.3e}")
    print(f"  |载入值 - Haar 初始值|   最大 = {(got - haar).abs().max():.3e}")

    if (got - tgt).abs().max() < 1e-6:
        print("  >>> 结论：载入正确，滤波器值被恢复。Q1 不是问题。")
    elif (got - haar).abs().max() < 1e-6:
        print("  >>> 结论：载入被忽略，推理时用的是纯 Haar！现有结果全部无效。")
    else:
        print("  >>> 结论：两者都不符合，需进一步排查。")

    # 同时报告 checkpoint 里的滤波器偏离 Haar 多远
    print(f"\n  checkpoint 滤波器偏离 Haar 的最大幅度: {(tgt - haar).abs().max():.4f}")
    return m_learn


# ---------------------------------------------------------------- Q2
def q2_roundtrip(model):
    print()
    print(SEP)
    print("Q2  DWT/IDWT 单元闭环误差（PR 性质）")
    print(SEP)

    torch.manual_seed(0)
    cases = {
        "随机图": torch.rand(1, 1, 256, 256),
        "常量图 0.5": torch.full((1, 1, 256, 256), 0.5),
        "脉冲图": torch.zeros(1, 1, 256, 256).index_put_(
            (torch.tensor([0]), torch.tensor([0]), torch.tensor([128]), torch.tensor([128])),
            torch.tensor(1.0),
        ),
        "奇数尺寸 255x255": torch.rand(1, 1, 255, 255),
    }

    for name, x in cases.items():
        # 奇数尺寸需先 pad 到偶数，因为 DWT 是 stride-2 卷积
        H, W = x.shape[2], x.shape[3]
        ph, pw = H % 2, W % 2
        xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        with torch.no_grad():
            bands = model.dwt(xp)
            rec = model.idwt(bands)
        rec = rec[:, :, :H, :W]
        diff = (rec - x).abs()
        max_err = diff.max().item()
        rel_l2 = (torch.norm(diff) / torch.norm(x)).item()
        print(
            f"  {name:18s}  max|err| = {max_err:.3e}   rel_L2 = {rel_l2:.3e}"
        )

    print("\n  对照：初始 Haar（未训练）的闭环")
    from models.dwt_layer import MultiLevelDWT, MultiLevelIDWT

    d0, i0 = MultiLevelDWT(learnable=False), MultiLevelIDWT(learnable=False)
    x = torch.rand(1, 1, 256, 256)
    with torch.no_grad():
        r = i0(d0(x))
    print(f"  {'Haar(未训练)':18s}  max|err| = {(r - x).abs().max():.3e}")

    print("\n  说明：max|err| 为最大绝对误差，rel_L2 为相对 L2 误差。")
    print("       PyTorch FP32 下，结构正确的闭环应在 1e-5 ~ 1e-6 量级。")


# ---------------------------------------------------------------- Q3
def q3_gate_bias(model):
    print()
    print(SEP)
    print("Q3  门控融合的 CT 偏置")
    print(SEP)

    ct_files = sorted(f for f in os.listdir(config.test_ct_dir) if f.endswith(".png"))
    ct_path = os.path.join(config.test_ct_dir, ct_files[0])
    mri_path = os.path.join(config.test_mri_dir, ct_files[0])

    ct = torch.from_numpy(load_img(ct_path))[None, None]
    mri = torch.from_numpy(load_img(mri_path))[None, None]
    H, W = ct.shape[2], ct.shape[3]
    ph, pw = (4 - H % 4) % 4, (4 - W % 4) % 4
    if ph or pw:
        ct = torch.nn.functional.pad(ct, (0, pw, 0, ph), mode="reflect")
        mri = torch.nn.functional.pad(mri, (0, pw, 0, ph), mode="reflect")

    print("  融合式: fused = ct + w*(mri-ct),   w = tanh(...)*0.4")
    print("  等价于 fused = (1-w)*ct + w*mri")
    print("  => CT 系数 (1-w) ∈ [0.6, 1.4] ；MRI 系数 w ∈ [-0.4, 0.4]\n")

    # ⚠️ 必须 eval()：模型含 24 个 BatchNorm2d 层，train() 模式会用批次统计
    #    而非训练时累积的 running statistics，导致门控权重完全失真。
    model.eval()
    with torch.no_grad():
        ct_b = model.dwt(ct)
        mri_b = model.dwt(mri)
        ws = []
        for name in model.fusion.band_names:
            fb = getattr(model.fusion, f"fuse_{name}")
            concat = torch.cat([ct_b[name], mri_b[name]], dim=1)
            w = fb.weight_net(concat) * 0.4
            ws.append(w.flatten())
        w_all = torch.cat(ws)

    q = [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    qs = torch.quantile(w_all, torch.tensor(q))
    print("  MRI 注入权重 w 的实际分布（7 个子带全部像素）：")
    for qi, vi in zip(q, qs):
        print(f"    分位 {qi:>5.2f} :  w = {vi:+.4f}")
    print(f"    min = {w_all.min():+.4f}   max = {w_all.max():+.4f}   mean = {w_all.mean():+.4f}")
    print(f"    w > 0.49（接近等权平均）占比: {(w_all > 0.49).float().mean()*100:.2f}%")
    print(f"    w < 0   （反向外推）     占比: {(w_all < 0).float().mean()*100:.2f}%")
    print("\n  注意：以上只反映融合块本身；后续 refine 与 EAR 可部分补偿。")


# ---------------------------------------------------------------- Q4
def q4_ssim_protocol(model):
    print()
    print(SEP)
    print("Q4  全图统计 SSIM  vs  局部窗口 SSIM")
    print(SEP)

    ct_files = sorted(f for f in os.listdir(config.test_ct_dir) if f.endswith(".png"))
    rows = []
    for f in ct_files:
        ct = load_img(os.path.join(config.test_ct_dir, f))
        mri = load_img(os.path.join(config.test_mri_dir, f))
        avg = (ct + mri) / 2.0

        ct_t = torch.from_numpy(ct)[None, None]
        mri_t = torch.from_numpy(mri)[None, None]
        model.eval()                      # 同上，必须 eval()
        with torch.no_grad():
            fused = model(ct_t, mri_t).squeeze().numpy()
        fused = (fused + 1.0) / 2.0 if fused.min() < 0 else fused
        fused = np.clip(fused, 0, 1)

        for method, img in [("Average", avg), ("MGF-Net", fused)]:
            g = (ssim_global(img, ct) + ssim_global(img, mri)) / 2
            l = (ssim_local(img, ct) + ssim_local(img, mri)) / 2
            rows.append((f, method, g, l))

    print(f"  {'图':<10}{'方法':<11}{'全图SSIM':>10}{'局部SSIM':>11}{'差值':>10}")
    print("  " + "-" * 52)
    for f, m, g, l in rows:
        print(f"  {f:<10}{m:<11}{g:>10.4f}{l:>11.4f}{g-l:>+10.4f}")

    print()
    for m in ["Average", "MGF-Net"]:
        gg = np.mean([r[2] for r in rows if r[1] == m])
        ll = np.mean([r[3] for r in rows if r[1] == m])
        print(f"  {m:<11} 全图口径均值 = {gg:.4f}   局部口径均值 = {ll:.4f}")

    ga = np.mean([r[2] for r in rows if r[1] == "Average"])
    gm = np.mean([r[2] for r in rows if r[1] == "MGF-Net"])
    la = np.mean([r[3] for r in rows if r[1] == "Average"])
    lm = np.mean([r[3] for r in rows if r[1] == "MGF-Net"])
    print()
    print(f"  全图口径下  Average - MGF-Net = {ga - gm:+.4f}")
    print(f"  局部口径下  Average - MGF-Net = {la - lm:+.4f}")


def main():
    print(SEP)
    print("MGF-Net Step-1 审计")
    print(SEP)
    model = q1_checkpoint_filters()
    q2_roundtrip(model)
    q3_gate_bias(model)
    q4_ssim_protocol(model)
    print()
    print(SEP)
    print("审计结束")
    print(SEP)


if __name__ == "__main__":
    main()
