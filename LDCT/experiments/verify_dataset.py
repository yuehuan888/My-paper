"""数据着陆自检：新下载的数据到底是不是原来那批。

为什么需要
================================================================
数据是重下的（官方 Box 不通，改用 Kaggle 镜像）。**镜像可能被重新打包、
可能少了患者、可能切片顺序不同** —— 而切片顺序会直接改变
`test_per_sample` 的对齐，进而改变一切逐切片分析。

不校验就开跑，等于把"数据换了"这个变量混进实验里。

校验什么
================================================================
不比对文件哈希（重打包后二进制必然不同），而是比对**可复现的数值指纹**：

  1. 患者数与每人切片数 —— 必须与 `identity_floors.json` 的 `n` 完全一致
  2. **identity 基线的 PSNR/SSIM** —— 这是最强判据：它只依赖
     (原始体数据, 划分, 评测口径)，与模型无关。
     已知值（口径 A，本仓库实测并写进论文）：
         L506  29.2489 / 0.8759
         L067  26.5576 / 0.7987
         L291  26.9805 / 0.8088
         S1 test 合并(L506+L067, 435 张) 27.8630 / 0.8361
     对不上就说明数据/划分/口径至少有一个不对，**不要继续跑实验**。

用法
================================================================
    python experiments/verify_dataset.py --h5 data/aapm_h5 \\
        --split splits/aapm_mayo_3mm.json

    # 只看逐患者切片数，不做完整评测（快）
    python experiments/verify_dataset.py --h5 data/aapm_h5 --counts-only
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# 已知的 identity 地板（来源：experiments/identity_floors.json，已写进论文 §4.4）
KNOWN = {
    "L143": (234, 25.4991, 0.7643),
    "L310": (214, 26.2577, 0.7273),
    "L067": (224, 26.5576, 0.7987),
    "L109": (128, 26.6052, 0.8165),
    "L096": (330, 26.9097, 0.7853),
    "L291": (343, 26.9805, 0.8088),
    "L333": (244, 27.2508, 0.8305),
    "L192": (240, 28.2896, 0.8344),
    "L286": (210, 29.1479, 0.8103),
    "L506": (211, 29.2489, 0.8759),
}
COMBINED_S1 = (435, 27.8630, 0.8361)   # L506 + L067

PSNR_TOL = 0.01     # dB —— 同数据同口径应当几乎逐位一致
SSIM_TOL = 5e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="aapm h5 目录")
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "aapm_mayo_3mm.json"))
    ap.add_argument("--counts-only", action="store_true")
    args = ap.parse_args()

    import h5py  # noqa: E402

    from data.dataset import make_dataset          # noqa: E402
    from experiments.train import evaluate         # noqa: E402

    print("=" * 74)
    print("数据着陆自检")
    print("=" * 74)
    print(f"h5 目录: {args.h5}")

    if not os.path.isdir(args.h5):
        sys.exit(f"FATAL: {args.h5} 不存在")

    # ---------------- 1. 逐患者切片数 ----------------
    print("\n[1] 逐患者切片数（对比 identity_floors.json）")
    fails = []
    # h5 布局：<h5>/<patient>_<dose>.h5 或 <h5>/<patient>.h5，两种都试
    counts = {}
    for pid in KNOWN:
        n = None
        for cand in (f"{pid}.h5", f"{pid}_quarter.h5", f"{pid}_full.h5"):
            p = os.path.join(args.h5, cand)
            if os.path.exists(p):
                with h5py.File(p, "r") as f:
                    # 取第一个数据集的长度
                    k = next(iter(f.keys()))
                    n = int(f[k].shape[0])
                break
        counts[pid] = n

    for pid, (exp_n, exp_p, exp_s) in KNOWN.items():
        got = counts[pid]
        ok = (got == exp_n)
        mark = "OK  " if ok else "FAIL"
        print(f"  [{mark}] {pid}  切片数 {got} (期望 {exp_n})")
        if got is not None and not ok:
            fails.append(f"{pid} 切片数 {got} != {exp_n}")

    if args.counts_only:
        print(f"\n结论：{'通过' if not fails else '失败'}")
        return 1 if fails else 0

    # ---------------- 2. identity 地板（最强判据） ----------------
    print("\n[2] identity 地板 PSNR/SSIM（口径 A）—— 模型无关，只依赖数据与口径")
    ds = make_dataset("aapm", "test", args.split)
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    rows, mean = evaluate(None, dl, "cpu")          # model=None -> identity

    print(f"  合并 (S1 test): PSNR={mean['PSNR']:.4f} (期望 {COMBINED_S1[1]:.4f})  "
          f"SSIM={mean['SSIM']:.4f} (期望 {COMBINED_S1[2]:.4f})")
    if abs(mean["PSNR"] - COMBINED_S1[1]) > PSNR_TOL:
        fails.append(f"合并 PSNR {mean['PSNR']:.4f} != {COMBINED_S1[1]:.4f}")
    if abs(mean["SSIM"] - COMBINED_S1[2]) > SSIM_TOL:
        fails.append(f"合并 SSIM {mean['SSIM']:.4f} != {COMBINED_S1[2]:.4f}")

    # 逐患者（按前 211 = L506、其后 = L067 的已知顺序）
    import numpy as np
    ps = np.array([r["PSNR"] for r in rows])
    seg = {"L506": ps[:211], "L067": ps[211:435]}
    for pid, arr in seg.items():
        exp_p = KNOWN[pid][1]
        print(f"  {pid}: PSNR={arr.mean():.4f} (期望 {exp_p:.4f})  n={len(arr)}")
        if abs(arr.mean() - exp_p) > PSNR_TOL:
            fails.append(f"{pid} PSNR {arr.mean():.4f} != {exp_p:.4f}")

    print()
    if fails:
        print("自检失败：")
        for f_ in fails:
            print("  -", f_)
        print("\n⚠️ 数据/划分/口径至少有一处不对，**不要继续跑实验**。")
        return 1
    print("自检通过 —— 数据与原始运行一致，可以开跑实验。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
