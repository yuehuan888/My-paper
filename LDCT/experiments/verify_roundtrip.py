"""
复现论文 §4.3 的机制表：三臂训练后的小波变换闭环误差。

论文里的原话
------------------------------------------------------------------
  | Arm           | Wavelet params | Round-trip error (rel. L2) |
  | pr            | 24             | 9.3e-08                    |
  | fixed         | 0              | 1.1e-07                    |
  | unconstrained | 64             | 9.9e-01                    |

  "Unconstrained learning drives the transform to a state where 99% of the
   signal is lost in a forward-backward pass."

**这是全文最有说服力、也最容易被攻击的一个数**：整个"可逆性被摧毁"的
机制论证都架在它上面。如果重跑不出来，论文的核心贡献就站不住。
故单独写这个脚本，用与 train.py **完全相同**的训练协议复现。

测量点：初始化时（未训练）与训练 50 步后各测一次——
"无约束学习**把**闭环误差推上去"是一个关于**变化**的主张，
所以必须同时给出初值，只报终值证明不了"学习摧毁了它"。

用法
------------------------------------------------------------------
    python experiments/verify_roundtrip.py
    python experiments/verify_roundtrip.py --steps 50 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from data.dataset import AapmDataset          # noqa: E402
from models.denoiser import PRWaveletDenoiser  # noqa: E402

AAPM_ROOT = os.path.join(ROOT, "data", "aapm_h5")
AAPM_SPLIT = os.path.join(ROOT, "splits", "aapm_mayo_3mm.json")

PAPER = {          # 论文 §4.3 声称的值，用于对账
    "pr": {"wavelet": 24, "rel": 9.3e-08},
    "fixed": {"wavelet": 0, "rel": 1.1e-07},
    "unconstrained": {"wavelet": 64, "rel": 9.9e-01},
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def probe(model, device, size=256):
    """在固定随机图上测闭环。固定种子 => 三臂测的是同一张图。"""
    g = torch.Generator(device="cpu").manual_seed(7)
    x = torch.rand(1, 1, size, size, generator=g).to(device)
    me, rel = model.transform_roundtrip(x)
    return me, rel


def from_checkpoints(device, prefix="w3_", n_seeds=5):
    """直接测**训练完成的 checkpoint** 的闭环误差。

    为什么必须做这一步：论文 §4.3 写的是"after 50 training steps"，
    但 50 步复现只得到 3.05e-02，与论文的 9.9e-01 差 32 倍。
    而论文的实验结果全部来自 w3_*/lit_* 这些**跑满 30 轮**的模型——
    若那些模型测出来确实是 ~1e-1，说明论文把"训练完成"写成了"50 步"，
    是**措辞问题**而非数值问题；若测出来仍是 ~1e-2，则数值本身有问题。
    """
    runs_dir = os.path.join(HERE, "runs")
    print(f"\n{'=' * 78}\n从训练完成的 checkpoint 复测闭环（{prefix}，论文结果的实际来源）\n"
          f"{'=' * 78}", flush=True)
    print(f"{'臂':<16}{'种子':>6}{'闭环 rel_L2':>16}", flush=True)
    print("-" * 78, flush=True)

    out = {}
    for arm in ("pr", "fixed", "unconstrained"):
        vals = []
        for s in range(n_seeds):
            ck = os.path.join(runs_dir, f"{prefix}{arm}_s{s}", "best.pth")
            if not os.path.exists(ck):
                continue
            m = PRWaveletDenoiser(levels=2, mid_ch=16, n_conv=2,
                                  wavelet=arm).to(device)
            st = torch.load(ck, map_location=device, weights_only=False)
            m.load_state_dict(st["model_state_dict"])
            m.eval()
            _, rel = probe(m, device)
            vals.append(rel)
            print(f"{arm:<16}{s:>6}{rel:>16.3e}", flush=True)
        if vals:
            mean = sum(vals) / len(vals)
            out[arm] = {"per_seed": vals, "mean": mean, "n": len(vals)}
            print(f"{arm:<16}{'均值':>6}{mean:>16.3e}   论文值 "
                  f"{PAPER[arm]['rel']:.1e}", flush=True)
    print("-" * 78, flush=True)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--checkpoints", action="store_true",
                    help="改为直接测训练好的 checkpoint（不重新训练）")
    ap.add_argument("--prefix", default="w3_")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoints:
        out = from_checkpoints(device, args.prefix, args.n_seeds)
        dst = os.path.join(HERE, f"roundtrip_checkpoints_{args.prefix.strip('_')}.json")
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n已写入 {dst}", flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    tr = AapmDataset(AAPM_ROOT, AAPM_SPLIT, "train", patch_size=args.patch,
                     is_training=True, cache=True)
    loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    print(f"\n协议：L1 + Adam(1e-3) + autocast + clip_grad_norm(1.0)，"
          f"bs={args.batch_size} patch={args.patch} 训练 {args.steps} 步 "
          f"seed={args.seed}", flush=True)
    print(f"{'臂':<16}{'小波参数':>9}{'闭环(初)':>12}{'闭环(训练后)':>14}"
          f"{'论文值':>12}{'判定':>8}", flush=True)
    print("-" * 78, flush=True)

    out = {}
    for arm in ("pr", "fixed", "unconstrained"):
        set_seed(args.seed)
        model = PRWaveletDenoiser(levels=2, mid_ch=16, n_conv=2,
                                  wavelet=arm).to(device)
        rep = model.param_report()

        me0, rel0 = probe(model, device)

        model.train()
        crit = nn.L1Loss()
        opt = optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
        it = iter(loader)
        for _ in range(args.steps):
            try:
                obs, gt = next(it)
            except StopIteration:
                it = iter(loader)
                obs, gt = next(it)
            obs, gt = obs.to(device), gt.to(device)
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = crit(model(obs), gt)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss = crit(model(obs), gt)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        me1, rel1 = probe(model, device)

        pw = PAPER[arm]
        # 判据：同一数量级即可（这类量的"精确复现"不现实，协议细节会带来一个量级的浮动）
        if rel1 <= 0:
            verdict = "?"
        else:
            ratio = rel1 / pw["rel"]
            verdict = "✅" if 0.1 <= ratio <= 10 else "⚠️"

        print(f"{arm:<16}{rep['wavelet']:>9}{rel0:>12.3e}{rel1:>14.3e}"
              f"{pw['rel']:>12.3e}{verdict:>8}", flush=True)

        out[arm] = {
            "wavelet_params": rep["wavelet"],
            "roundtrip_init_relL2": rel0,
            "roundtrip_after_relL2": rel1,
            "roundtrip_init_maxabs": me0,
            "roundtrip_after_maxabs": me1,
            "paper_wavelet_params": pw["wavelet"],
            "paper_relL2": pw["rel"],
            "seed": args.seed,
            "steps": args.steps,
        }

    print("-" * 78, flush=True)
    print("\n判读：", flush=True)
    print("  · pr / fixed 两臂训练后应仍在 float32 极限附近（~1e-7）", flush=True)
    print("  · unconstrained 臂若确实被训练推到 ~1e-1（99% 信号丢失），", flush=True)
    print("    则 §4.3 的机制主张成立；若它仍停在 ~1e-7，全文核心论证不成立。", flush=True)

    dst = os.path.join(HERE, "verify_roundtrip.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {dst}", flush=True)


if __name__ == "__main__":
    main()
