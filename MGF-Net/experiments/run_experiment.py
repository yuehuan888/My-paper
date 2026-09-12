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
import csv
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
def build_index(split_path: str, manifest_path: str | None = None) -> dict:
    """构建 {id: (ct_path, mri_path)} 索引。

    优先用 manifest CSV（可追溯、含病例号）；没有则回退到扫描 data/train 与
    data/test——后者只适用于早期的小规模筛查数据。

    split 文件里可能同时含 splits（病例级）与 slice_level_random（切片级随机
    对照），用 --split-key 选择。
    """
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)

    index = {}
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                index[row["pair_id"]] = {
                    "ct": os.path.join(ROOT, row["path_a"]),
                    "mri": os.path.join(ROOT, row["path_b"]),
                    "case_id": int(row["case_id"]),
                    "slice_id": int(row["slice_id"]),
                }
        src = f"manifest:{os.path.relpath(manifest_path, ROOT)}"
    else:
        for sub in ["train", "test"]:
            for mod in ["ct", "mri"]:
                d = os.path.join(ROOT, "data", sub, mod)
                if not os.path.isdir(d):
                    continue
                for fn in os.listdir(d):
                    stem, ext = os.path.splitext(fn)
                    if ext.lower() in IMG_EXTS:
                        index.setdefault(stem, {})[mod] = os.path.join(d, fn)
        src = "扫描 data/train 与 data/test"

    for i, v in index.items():
        for p in (v.get("ct"), v.get("mri")):
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"ID {i} 的图片不存在: {p}")

    for k, v in split.get("splits", {}).items():
        missing = [i for i in v if i not in index]
        if missing:
            raise ValueError(f"划分 {k} 中的 ID 在数据中不存在: {missing[:10]}")

    return {"index": index, "split": split, "source": src}


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

    # LR milestones 按轮数**比例缩放**。
    # config 原硬编码 [40,70,90]，但只跑 50 轮时 70/90 永不触发——
    # 实测 86% 的运行在末 10 轮损失斜率仍为负（均值 −0.00083/轮），即未收敛。
    # 默认取 50%/80%/90%，与原配置在 100 轮时等价。
    if args.lr_milestones:
        milestones = [int(x) for x in args.lr_milestones.split(",")]
    else:
        milestones = [max(1, int(round(args.epochs * f))) for f in (0.5, 0.8, 0.9)]
        milestones = sorted(set(m for m in milestones if 0 < m < args.epochs))

    data = build_index(args.split, args.manifest)
    index, split = data["index"], data["split"]

    # split-key 决定用病例级划分还是"切片级随机"对照
    def block_of(key):
        if key == "slice_level_random":
            return split["slice_level_random"]["splits"]
        # 交叉验证划分：folds 是折列表，用 --fold 选折
        if "folds" in split and args.fold is not None:
            return split["folds"][args.fold]["splits"]
        return split["splits"]

    train_block = block_of(args.split_key)
    # eval-split 可与 train 不同：用于"用泄漏划分训练、在干净留出集上评价"，
    # 直接量化数据泄漏带来的虚高。默认与 split-key 一致。
    eval_block = block_of(args.eval_split)

    tr_ids = train_block["train"]
    va_ids = train_block.get("val", [])
    te_ids = eval_block["test"]

    run_dir = os.path.join(HERE, "runs", args.tag)
    os.makedirs(os.path.join(run_dir, "fused"), exist_ok=True)

    cfg = {
        "tag": args.tag,
        "label": args.label,
        "created": datetime.now(timezone.utc).isoformat(),
        "code_version": code_version(),
        "seed": args.seed,
        "epochs": args.epochs,
        "gate_type": args.gate,
        "wavelet": args.wavelet,
        "learnable_dwt": bool(args.learnable_dwt),
        "mid_channels": args.mid_channels,
        "patch_size": args.patch_size,
        "batch_size": args.batch_size,
        "oversample": args.oversample,
        "learning_rate": args.learning_rate,
        "lr_milestones": milestones,
        "loss": {"alpha_ssim": args.alpha, "beta_l1": args.beta,
                 "gamma_grad": args.gamma, "delta_balance": args.delta,
                 "grad_mode": args.grad_mode,
                 "grad_excess_weight": args.grad_excess_weight},
        "use_edge_refine": args.edge_refine,
        "split_file": os.path.relpath(args.split, ROOT),
        "split_key": args.split_key,
        "eval_split": args.eval_split,
        "fold": args.fold,
        "split_ids": {"train": tr_ids, "val": va_ids, "test": te_ids},
        "split_cases": {
            "train": sorted({index[i].get("case_id") for i in tr_ids} - {None}),
            "val": sorted({index[i].get("case_id") for i in va_ids} - {None}),
            "test": sorted({index[i].get("case_id") for i in te_ids} - {None}),
        },
        "index_source": data["source"],
        "manifest": (os.path.relpath(args.manifest, ROOT)
                     if os.path.exists(args.manifest) else None),
        "select_metric": args.select_metric,
        "device": str(device),
        "torch": torch.__version__,
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("=" * 74)
    print(f"实验 {args.tag}")
    print(f"  门控      : {args.gate}")
    print(f"  频域变换  : {args.wavelet}")
    print(f"  LR衰减点  : {milestones}  (共 {args.epochs} 轮)")
    print(f"  梯度损失  : {args.grad_mode}"
          + (f" (excess_weight={args.grad_excess_weight})" if args.grad_mode=="hinge" else ""))
    print(f"  edge_refine: {'开启' if args.edge_refine else '关闭（默认）'}"
          f"    balance权重: {args.delta}   梯度权重: {args.gamma}")
    print(f"  索引来源  : {data['source']}")
    print(f"  划分方式  : {args.split_key}" +
          (f"  折={args.fold}" if args.fold is not None else ""))
    print(f"  种子      : {args.seed}   轮数: {args.epochs}   patch: {args.patch_size}")
    print(f"  规模      : train={len(tr_ids)}  val={len(va_ids)}  test={len(te_ids)}")
    print(f"  训练病例  : {cfg['split_cases']['train']}")
    print(f"  测试病例  : {cfg['split_cases']['test']}")
    if set(cfg["split_cases"]["train"]) & set(cfg["split_cases"]["test"]):
        print(f"  ⚠️ 训练与测试共享病例: {sorted(set(cfg['split_cases']['train']) & set(cfg['split_cases']['test']))}")
    print(f"  选模指标  : {args.select_metric}（基于验证集）")
    print(f"  代码版本  : {cfg['code_version']}")
    print("=" * 74)

    # 训练集：按 manifest 显式路径对构建
    train_ds = MedicalFusionDataset(
        mode="dir", patch_size=args.patch_size, is_training=True,
        oversample=args.oversample,
        pairs=[(index[i]["ct"], index[i]["mri"]) for i in tr_ids],
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)

    model = MGFNet(in_channels=1, mid_channels=args.mid_channels,
                   learnable_dwt=bool(args.learnable_dwt),
                   gate_type=args.gate, wavelet=args.wavelet,
                   use_edge_refine=args.edge_refine).to(device)
    n_params = count_parameters(model)

    criterion = MGFusionLoss(alpha=args.alpha, beta=args.beta,
                             gamma=args.gamma, delta=args.delta,
                             grad_mode=args.grad_mode,
                             grad_excess_weight=args.grad_excess_weight).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=config.lr_decay)
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
            "wavelet": args.wavelet,
            "learnable_dwt": bool(args.learnable_dwt),
            "mid_channels": args.mid_channels,
            "model_state_dict": best["state"],
            "config": cfg,
        }, ckpt_path)
    else:
        torch.save({"epoch": args.epochs, "gate_type": args.gate,
                    "wavelet": args.wavelet,
                    "model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    # 用验证集选出的最优权重评价
    results = {"tag": args.tag, "gate_type": args.gate, "wavelet": args.wavelet,
               "seed": args.seed,
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

    # 训练后实测变换闭环（PR-LWT 应给出 ~1e-7；legacy 会显著更差）
    try:
        xt = to_tensor(load_u8(index[te_ids[0]]["ct"]), device)
        max_err, rel_l2 = model.transform_roundtrip(xt)
        results["transform_roundtrip"] = {"max_abs_err": max_err, "rel_l2": rel_l2}
        print(f"变换闭环  : max|err|={max_err:.3e}  rel_L2={rel_l2:.3e}  ({args.wavelet})")
    except Exception as e:  # noqa: BLE001
        print(f"变换闭环测量失败: {e}")

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
    ap.add_argument("--label", default=None,
                    help="实验条件标签，用于跨种子分组汇总（默认取 tag 去掉 _s<数字> 后缀）")
    ap.add_argument("--gate", default="residual_capped",
                    choices=["residual_capped", "neutral_sigmoid"])
    ap.add_argument("--wavelet", default="legacy", choices=["legacy", "lifting"],
                    help="legacy=旧的分离式可学习滤波器（无闭环保证）；"
                         "lifting=PR-LWT（提升格式，可逆性由结构保证）")
    ap.add_argument("--split", default=os.path.join(ROOT, "splits", "ct_mri_screen_v1.json"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest_ct_mri.csv"),
                    help="逐图清单 CSV；存在则优先使用（含病例号）")
    ap.add_argument("--fold", type=int, default=None,
                    help="交叉验证划分文件中选第几折（该文件须含 folds 字段）。"
                         "见 data/make_cv_splits.py")
    ap.add_argument("--split-key", default="splits",
                    choices=["splits", "slice_level_random"],
                    help="训练用划分：splits=病例级；slice_level_random=切片级随机（泄漏对照）")
    ap.add_argument("--eval-split", default=None,
                    choices=[None, "splits", "slice_level_random"],
                    help="评价用测试集，默认与 --split-key 相同。"
                         "设为 splits 可让泄漏模型在干净留出集上评价，用于量化泄漏")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--learnable-dwt", type=int, default=1, choices=[0, 1])
    ap.add_argument("--mid-channels", type=int, default=32)
    ap.add_argument("--patch-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--oversample", type=int, default=25)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--grad-mode", default="abs", choices=["abs", "hinge"],
                    help="梯度损失模式。abs=原式 |∇F−max|，要求处处取最强边，"
                         "数学上不可能同时忠实于两个源（实测为保真度差距主因）；"
                         "hinge=只重罚'漏掉的边'，允许适度超出")
    ap.add_argument("--grad-excess-weight", type=float, default=0.25,
                    help="hinge 模式下对'多出的边'的惩罚权重")
    ap.add_argument("--lr-milestones", default=None,
                    help="逗号分隔的 epoch 数；默认按轮数比例取 50%%/80%%/90%%")
    ap.add_argument("--alpha", type=float, default=1.0, help="SSIM 损失权重")
    ap.add_argument("--beta", type=float, default=10.0, help="L1 损失权重")
    ap.add_argument("--gamma", type=float, default=5.0, help="梯度损失权重")
    ap.add_argument("--delta", type=float, default=2.0,
                    help="balance 损失权重（|L1(f,CT)-L1(f,MRI)|）。"
                         "注意：像素平均使该项恰为 0，即平均是该权重的精确最优解")
    ap.add_argument("--edge-refine", action="store_true",
                    help="启用 edge_refine 模块（**默认关闭**）。"
                         "实测：该模块占 20,000 参数（21%%），但旁路它对全部十项指标"
                         "无任何可测影响（MI 2.3999→2.3920、Qabf 0.6139→0.6088 均在噪声内）。"
                         "故默认删除，模型收敛为 小波分解 → 逐子带门控融合 → 跨频段协调 → 重建。"
                         "保留开关仅用于消融表。")
    ap.add_argument("--select-metric", default="SSIM", choices=METRIC_KEYS,
                    help="据验证集选模用的指标")
    ap.add_argument("--val-interval", type=int, default=5,
                    help="每多少 epoch 跑一次验证")
    args = ap.parse_args()
    import re as _re
    if args.label is None:
        args.label = _re.sub(r"_s\d+$", "", args.tag)
    if args.eval_split is None:
        args.eval_split = args.split_key
    run(args)


if __name__ == "__main__":
    main()
