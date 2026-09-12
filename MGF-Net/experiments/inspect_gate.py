"""
检查已训练模型的门控权重分布。

回答一个机制问题：门控网络实际学到的 MRI 注入权重 w 覆盖了多大范围？
如果 w 挤在某个极值附近（而不是铺开），说明融合退化成"整图偏向某一模态"，
而不是"按区域自适应选择"——这是审计 Q3 的核心关切。

用法：
    python experiments/inspect_gate.py experiments/runs/<tag>/best.pth [--dataset test]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models.mgf_net import MGFNet  # noqa: E402


def load_u8(p):
    return np.array(Image.open(p).convert("L"), dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--dataset", default="test", choices=["test", "train"])
    ap.add_argument("--ids", default=None, help="逗号分隔，默认该目录全部")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    gate = ck.get("gate_type", cfg.get("gate_type", "residual_capped"))
    mid = ck.get("mid_channels", cfg.get("mid_channels", 32))
    ldwt = ck.get("learnable_dwt", cfg.get("learnable_dwt", True))

    model = MGFNet(1, mid, learnable_dwt=ldwt, gate_type=gate)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()          # 必须：24 个 BN 层，train() 会给出完全失真的结果

    ids = args.ids.split(",") if args.ids else sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(os.path.join(ROOT, "data", args.dataset, "ct"))
        if f.lower().endswith(".png")
    )

    print("=" * 74)
    print(f"门控权重分布检查")
    print(f"  checkpoint : {os.path.basename(os.path.dirname(args.ckpt))}/{os.path.basename(args.ckpt)}")
    print(f"  gate_type  : {gate}")
    print(f"  训练 epoch : {ck.get('epoch')}   val_score: {ck.get('val_score')}")
    print("=" * 74)
    print(f"  {'图':<8}{'min':>9}{'q25':>9}{'中位':>9}{'q75':>9}{'max':>9}{'均值':>9}{'标准差':>9}")
    print("  " + "-" * 62)

    all_w = []
    for i in ids:
        ct = load_u8(os.path.join(ROOT, "data", args.dataset, "ct", f"{i}.png"))
        mri = load_u8(os.path.join(ROOT, "data", args.dataset, "mri", f"{i}.png"))
        ct_t = torch.from_numpy(ct.astype(np.float32) / 255)[None, None]
        mri_t = torch.from_numpy(mri.astype(np.float32) / 255)[None, None]
        H, W = ct_t.shape[2], ct_t.shape[3]
        ph, pw = (4 - H % 4) % 4, (4 - W % 4) % 4
        if ph or pw:
            ct_t = torch.nn.functional.pad(ct_t, (0, pw, 0, ph), mode="reflect")
            mri_t = torch.nn.functional.pad(mri_t, (0, pw, 0, ph), mode="reflect")

        with torch.no_grad():
            ct_b, mri_b = model.dwt(ct_t), model.dwt(mri_t)
            ws = []
            for name in model.fusion.band_names:
                fb = getattr(model.fusion, f"fuse_{name}")
                ws.append(fb.gate_weights(ct_b[name], mri_b[name]).flatten())
            w = torch.cat(ws).numpy()

        all_w.append(w)
        q = np.percentile(w, [0, 25, 50, 75, 100])
        print(f"  {i:<8}{q[0]:>9.4f}{q[1]:>9.4f}{q[2]:>9.4f}{q[3]:>9.4f}{q[4]:>9.4f}"
              f"{w.mean():>9.4f}{w.std():>9.4f}")

    w = np.concatenate(all_w)
    print("  " + "-" * 62)
    print(f"\n  全部像素（{w.size:,} 个）:")
    lo, hi = (0.0, 1.0) if gate == "neutral_sigmoid" else (-0.4, 0.4)
    near_lo = float(np.mean(w < lo + 0.02 * (hi - lo)))
    near_hi = float(np.mean(w > hi - 0.02 * (hi - lo)))
    print(f"    值域 [{w.min():+.4f}, {w.max():+.4f}]   理论范围 [{lo}, {hi}]")
    print(f"    贴下限(±2%) 占比: {near_lo*100:6.2f}%")
    print(f"    贴上限(±2%) 占比: {near_hi*100:6.2f}%")
    print(f"    w > 0.49（跨过等权平均）占比: {float(np.mean(w > 0.49))*100:6.2f}%")
    print(f"    标准差: {w.std():.4f}")

    print("\n  判读：")
    if near_lo + near_hi > 0.5:
        print("    ⚠️ 超过一半像素贴在值域边界 —— 门控接近饱和，")
        print("       融合退化为'整图偏向单一模态'而非'按区域自适应选择'。")
    elif w.std() < 0.02:
        print("    ⚠️ 权重几乎无空间变化 —— 门控退化成常数，失去自适应能力。")
    else:
        print("    ✅ 权重在值域内铺开，未饱和，具备空间自适应能力。")


if __name__ == "__main__":
    main()
