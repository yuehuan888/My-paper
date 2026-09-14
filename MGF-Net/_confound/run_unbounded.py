"""Confound check: PR-LWT with the tanh bound effectively removed.

Identical to experiments/run_experiment.py --wavelet lifting --learnable-dwt 1,
except bank.bound is raised from 0.5 to 10.0 (tanh never saturates -> unbounded
in practice) after construction.  Everything else (split, seed, loss, lr,
schedule, AMP, clipping, val interval, select metric) is the frozen recipe.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "experiments"))
import run_experiment as R
from config import config
from data.dataset import MedicalFusionDataset
from losses import MGFusionLoss
from models.mgf_net import MGFNet, count_parameters
from models.lifting_dwt import _LiftingBank1D

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--bound", type=float, default=10.0)
a = ap.parse_args()

R.set_seed(a.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = R.build_index(os.path.join(ROOT, "splits", "ct_mri_case_v1.json"),
                     os.path.join(ROOT, "data", "manifest_ct_mri.csv"))
index, split = data["index"], data["split"]
tr_ids = split["splits"]["train"]; va_ids = split["splits"]["val"]; te_ids = split["splits"]["test"]

run_dir = os.path.join(ROOT, "experiments", "runs", a.tag)
os.makedirs(os.path.join(run_dir, "fused"), exist_ok=True)

train_ds = MedicalFusionDataset(mode="dir", patch_size=128, is_training=True, oversample=3,
                                pairs=[(index[i]["ct"], index[i]["mri"]) for i in tr_ids])
train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

model = MGFNet(in_channels=1, mid_channels=32, learnable_dwt=True,
               gate_type="neutral_sigmoid", wavelet="lifting").to(device)
n_over = 0
for m in model.modules():
    if isinstance(m, _LiftingBank1D):
        m.bound = float(a.bound); n_over += 1
print(f"[confound] overrode bound={a.bound} on {n_over} banks", flush=True)
print(f"[confound] starting bound={a.bound} seed={a.seed} epochs={a.epochs} "
      f"train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}", flush=True)

n_params = count_parameters(model)
criterion = MGFusionLoss(alpha=1.0, beta=10.0, gamma=5.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_decay_epochs, gamma=config.lr_decay)
scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

history = {"epoch": [], "train_loss": [], "val": []}
best = {"score": -np.inf, "epoch": None, "state": None, "val_rows": None}
t0 = time.time()
for epoch in range(1, a.epochs + 1):
    model.train(); losses = []
    for ct, mri in train_dl:
        ct, mri = ct.to(device), mri.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda"):
            fused = model(ct, mri); loss, _ = criterion(fused, ct, mri)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        losses.append(loss.item())
    scheduler.step()
    tl = float(np.mean(losses))
    history["epoch"].append(epoch); history["train_loss"].append(tl)
    if epoch % 5 == 0 or epoch == a.epochs:
        rows = R.evaluate_split(model, va_ids, index, device)
        score = float(np.mean([r["SSIM"] for r in rows]))
        history["val"].append({"epoch": epoch, "SSIM": score, **R.mean_metrics(rows)})
        if score > best["score"]:
            best.update(score=score, epoch=epoch, val_rows=rows,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        print(f"  epoch {epoch:3d} | loss {tl:.4f} | valSSIM {score:.4f}"
              f"{'  *best*' if best['epoch']==epoch else ''}", flush=True)
    else:
        print(f"  epoch {epoch:3d} | loss {tl:.4f}", flush=True)
wall = time.time() - t0

torch.save({"epoch": best["epoch"], "val_score": best["score"], "bound": a.bound,
            "model_state_dict": best["state"]}, os.path.join(run_dir, "best.pth"))
if best["state"] is not None:
    model.load_state_dict(best["state"])
results = {"tag": a.tag, "wavelet": "lifting", "bound": a.bound, "seed": a.seed,
           "best_epoch": best["epoch"], "best_val_score": best["score"],
           "n_params": n_params, "wall_seconds": wall}
results["val"] = {"per_image": best["val_rows"], "mean": R.mean_metrics(best["val_rows"])}
te_rows = R.evaluate_split(model, te_ids, index, device)
results["test"] = {"per_image": te_rows, "mean": R.mean_metrics(te_rows)}
try:
    xt = R.to_tensor(R.load_u8(index[te_ids[0]]["ct"]), device)
    me, rl = model.transform_roundtrip(xt)
    results["transform_roundtrip"] = {"max_abs_err": me, "rel_l2": rl}
    print(f"[confound] roundtrip rel_L2={rl:.3e}", flush=True)
except Exception as e:
    print("roundtrip failed", e)
with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)
print("[confound] TEST MEAN:", json.dumps(results["test"]["mean"]), flush=True)
print(f"[confound] DONE best_epoch={best['epoch']} wall={wall/60:.2f}min", flush=True)
