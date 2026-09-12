"""
单个实验的运行器。

设计目标（对应《发表路线与实验计划_修订版》§四）：
  1. **完全可复现** —— 固定随机种子，保存完整配置、划分与代码版本
  2. **按验证集选模** —— 不拿训练损失当 best（原 `train.py` 的做法）
  3. **测试集只评一次** —— 方案冻结后才碰 test
  4. **指标用 `utils.metrics`** —— 局部窗口 SSIM 等标准实现，
     不用 `test.py` 里那个全图统计的退化版本

用法：
    python experiments/run_experiment.py \
        --tag gate_neutral_seed0 \
        --gate neutral_sigmoid \
        --split splits/ct_mri_screen_v1.json \
        --seed 0 --epochs 100

输出到 experiments/runs/<tag>/：
    config.json    完整配置（含划分与代码版本）
    results.json   验证集与测试集的逐图指标
    history.json   每 epoch 的损失与验证指标
    fused/         验证集与测试集的融合结果图
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from config import config                              # noqa: E402
from data.dataset import MedicalFusionDataset          # noqa: E402
from losses import MGFusionLoss                        # noqa: E402
from models.mgf_net import MGFNet, count_parameters    # noqa: E402
from utils.metrics import evaluate_pair                # noqa: E402

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
METRIC_KEYS = ["EN", "SD", "SF", "MI", "CC", "PSNR", "SSIM", "VIF", "SCD", "Qabf"]


# ------------------------------------------------------------------ 可复现性
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def code_version() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        )
        rev = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
        return f"{rev[:12]}{'-dirty' if dirty else ''}"
    except Exception:
        return "unavailable"


def dir_hash(path: str) -> str:
    """目录内容的哈希，用于记录"这份结果对应哪份数据"。"""
    h = hashlib.sha256()
    if not os.path.isdir(path):
        return "missing"
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if os.path.isfile(p) and f.lower().endswith(IMG_EXTS):
            h.update(f.encode())
            h.update(str(os.path.getsize(p)).encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ 数据索引
def build_index(split_path: str) -> dict:
    """把 data/train 与 data/test 合并成一个 {id: (ct_path, mri_path)} 视图。

    当前数据分散在两个目录，而划分是按 ID 指定的，所以需要合并后按 ID 取。
    """
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)

    index = {}
    for sub in ["train", "test"]:
        for mod in ["ct", "mri"]:
            d = os.path.join(ROOT, "data", sub, mod)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                stem, ext = os.path.splitext(fn)
                if ext.lower() not in IMG_EXTS:
                    continue
                index.setdefault(stem, {})[mod] = os.path.join(d, fn)

    ids = sorted(index)
    incomplete = [i for i in ids if "ct" not in index[i] or "mri" not in index[i]]
    if incomplete:
        raise ValueError(f"以下 ID 缺 ct 或 mri: {incomplete}")

    # 校验划分里提到的 ID 都存在
    for k, v in split["splits"].items():
        missing = [i for i in v if i not in index]
        if missing:
            raise ValueError(f"划分 {k} 中的 ID 在数据中不存在: {missing}")

    return {"index": index, "split": split, "all_ids": ids}


def load_u8(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def to_tensor(img_u8: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(img_u8.astype(np.float32) / 255.0)[None, None]
    H, W = t.shape[2], t.shape[3]
    ph, pw = (4 - H % 4) % 4, (4 - W % 4) % 4
    if ph or pw:
        t = torch.nn.functional.pad(t, (0, pw, 0, ph), mode="reflect")
    return t.to(device)


@torch.no_grad()
def infer_full(model, ct_u8, mri_u8, device):
    """全图推理，返回 [0,1] 的 float 融合图（裁剪回原尺寸）。

    ⚠️ 必须 model.eval()——模型含 24 个 BatchNorm2d 层，
    train() 模式会用批次统计而非 running statistics，输出完全失真。
    """
    model.eval()
    ct_t, mri_t = to_tensor(ct_u8, device), to_tensor(mri_u8, device)
    out = model(ct_t, mri_t).squeeze().float().cpu().numpy()
    return out[: ct_u8.shape[0], : ct_u8.shape[1]]


def evaluate_split(model, ids, index, device):
    """在给定 ID 列表上算全部指标，返回逐图结果。"""
    rows = []
    for i in ids:
        ct = load_u8(index[i]["ct"])
        mri = load_u8(index[i]["mri"])
        fused01 = infer_full(model, ct, mri, device)
        fused_u8 = np.clip(np.rint(fused01 * 255.0), 0, 255).astype(np.uint8)
        m = evaluate_pair(fused_u8, ct, mri)
        row = {"id": i}
        row.update({k: float(m[k]) for k in METRIC_KEYS})
        rows.append(row)
    return rows


def mean_metrics(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}


# ------------------------------------------------------------------ 主流程
def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_index(args.split)
    index, split = data["index"], data["split"]
    tr_ids = split["splits"]["train"]
    va_ids = split["splits"].get("val", [])
    te_ids = split["splits"]["test"]

    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(os.path.join(run_dir, "fused"), exist_ok=True)

    cfg = {
        "tag": args.tag,
        "created": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "seed": args.seed,
        "epochs": args.epochs,
        "gate_type": args.gate,
        "learnable_dwt": bool(args.learnable_dwt),
        "mid_channels": args.mid_channels,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "oversample": args.oversample,
        "learning_rate": args.learning_rate,
        "loss": {"alpha_ssim": args.alpha, "beta_l1": args.beta, "gamma_grad": args.gamma},
        "split_file": os.path.relpath(args.split, ROOT),
        "split_ids": {"train": tr_ids, "val": va_ids, "test": te_ids},
        "select_metric": args.select_metric,
        "data_hash": {"data/train/ct": dir_hash(os.path.join(ROOT, "data/train/ct")),
                      "data/test/ct": dir_hash(os.path.join(ROOT, "data/test/ct"))},
        "device": str(device),
        "torch": torch.__version__,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("=" * 74)
    print(f"实验 {args.tag}")
    print(f"  门控      : {args.gate}")
    print(f"  种子      : {args.seed}   轮数: {args.epochs}   patch: {args.patch_size}")
    print(f"  划分      : train={tr_ids}  val={va_ids}  test={te_ids}")
    print(f"  选模指标  : {args.select_metric}（基于验证集）")
    print(f"  代码版本  : {cfg['code_version']}")
    print("=" * 74)

    # 训练集
    train_ds = MedicalFusionDataset(
        data_path=os.path.join(ROOT, "data", "train"), mode="dir",
        patch_size=args.patch_size, is_training=True,
        oversample=args.oversample, ids=tr_ids,
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = MGFNet(in_channels=1, mid_channels=args.mid_channels,
                   learnable_dwt=bool(args.learnable_dwt),
                   gate_type=args.gate).to(device)
    n_params = count_parameters(model)

    criterion = MGFusionLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_decay_epochs, gamma=config.lr_decay)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"epoch": [], "train_loss": [], "val": []}
    best = {"score": -np.inf, "epoch": None, "state": None, "val_rows": None}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for ct, mri in train_dl:
            ct, mri = ct.to(device), mri.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    fused = model(ct, mri)
                    loss, _ = criterion(fused, ct, mri)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                fused = model(ct, mri)
                loss, _ = criterion(fused, ct, mri)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        train_loss = float(np.mean(losses))
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)

        # 按验证集选模（不是训练损失）
        val_rows = None
        if va_ids and (epoch % args.val_interval == 0 or epoch == args.epochs):
            val_rows = evaluate_split(model, va_ids, index, device)
            score = float(np.mean([r[args.select_metric] for r in val_rows]))
            history["val"].append({"epoch": epoch,
                                   args.select_metric: score,
                                   **mean_metrics(val_rows)})
            if score > best["score"]:
                best.update(score=score, epoch=epoch, val_rows=val_rows,
                            state={k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()})
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                  f"val {args.select_metric} {score:.4f}"
                  f"{'  *best*' if best['epoch'] == epoch else ''}")
        else:
            print(f"  epoch {epoch:3d} | train_loss {train_loss:.4f}")

    wall = time.time() - t0

    # 保存最优权重（按验证集选出的那个 epoch）。
    # 没有它就无法复现、无法事后诊断（例如检查门控权重的实际分布）。
    ckpt_path = os.path.join(run_dir, "best.pth")
    if best["state"] is not None:
        torch.save({
            "epoch": best["epoch"],
            "val_score": best["score"],
            "select_metric": args.select_metric,
            "gate_type": args.gate,
            "learnable_dwt": bool(args.learnable_dwt),
            "mid_channels": args.mid_channels,
            "model_state_dict": best["state"],
            "config": cfg,
        }, ckpt_path)
    else:
        torch.save({"epoch": args.epochs, "gate_type": args.gate,
                    "model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    # 用验证集选出的最优权重评价
    results = {"tag": args.tag, "gate_type": args.gate, "seed": args.seed,
               "best_epoch": best["epoch"], "best_val_score": best["score"],
               "n_params": n_params, "wall_seconds": wall,
               "checkpoint": os.path.relpath(ckpt_path, ROOT)}
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    if va_ids:
        results["val"] = {"per_image": best["val_rows"] if best["val_rows"] else
                          evaluate_split(model, va_ids, index, device)}
        results["val"]["mean"] = mean_metrics(results["val"]["per_image"])

    # 测试集：仅此一次
    test_rows = evaluate_split(model, te_ids, index, device)
    results["test"] = {"per_image": test_rows, "mean": mean_metrics(test_rows)}

    # 存融合图，便于人工核对
    for i in te_ids:
        f = infer_full(model, load_u8(index[i]["ct"]), load_u8(index[i]["mri"]), device)
        Image.fromarray(np.clip(np.rint(f * 255), 0, 255).astype(np.uint8)).save(
            os.path.join(run_dir, "fused", f"test_{i}.png"))

    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("-" * 74)
    print(f"最优 epoch = {best['epoch']}（验证集 {args.select_metric} = {best['score']:.4f}）")
    print(f"参数量 = {n_params:,}   用时 = {wall/60:.2f} min")
    print(f"{'':<12}" + "".join(f"{k:>9}" for k in METRIC_KEYS))
    tm = results["test"]["mean"]
    print(f"{'TEST':<12}" + "".join(f"{tm[k]:>9.4f}" for k in METRIC_KEYS))
    if va_ids:
        vm = results["val"]["mean"]
        print(f"{'VAL':<12}" + "".join(f"{vm[k]:>9.4f}" for k in METRIC_KEYS))
    print(f"\n结果目录: {run_dir}")
    return results


def main():
    ap = argparse.ArgumentParser(description="MGF-Net 单实验运行器")
    ap.add_argument("--tag", required=True, help="实验标识，输出到 experiments/runs/<tag>")
    ap.add_argument("--gate", default="residual_capped",
                    choices=["residual_capped", "neutral_sigmoid"])
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "ct_mri_screen_v1.json"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learnable-dwt", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mid-channels", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--oversample", type=int, default=25)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0, help="SSIM 损失权重")
    ap.add_argument("--beta", type=float, default=10.0, help="L1 损失权重")
    ap.add_argument("--gamma", type=float, default=5.0, help="梯度损失权重")
    ap.add_argument("--select-metric", default="SSIM", choices=METRIC_KEYS,
                    help="据验证集选模用的指标")
    ap.add_argument("--val-interval", type=int, default=5,
                    help="每多少 epoch 跑一次验证")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
