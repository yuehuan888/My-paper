"""临时核查 2：β=0.5 到底是 n=3 还是 n=10？以及 E1 容量表的正确取数。"""

from __future__ import annotations

import glob
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


def load(tag):
    d = os.path.join(HERE, "runs", tag)
    cfg_p, res_p, ck_p = (os.path.join(d, f) for f in
                          ("config.json", "results.json", "best.pth"))
    if not all(os.path.exists(p) for p in (cfg_p, res_p, ck_p)):
        return None
    cfg = json.load(io.open(cfg_p, encoding="utf-8"))
    res = json.load(io.open(res_p, encoding="utf-8"))
    st = torch.load(ck_p, map_location="cpu", weights_only=False)
    sd = st.get("model") or st.get("model_state_dict")
    if any(k.startswith("dwt.") for k in sd):
        return {"tag": tag, "old_arch": True, "cfg": cfg, "psnr": res["test_mean"]["PSNR"]}
    m = PRWaveletDenoiser(bound=cfg["bound"], init_drift=cfg.get("init_drift", 0.0),
                          n_conv=cfg.get("n_conv", 2))
    m.load_state_dict(sd)
    return {
        "tag": tag, "old_arch": False, "cfg": cfg,
        "bound": cfg["bound"], "n_conv": cfg.get("n_conv", 2),
        "n_params": cfg.get("n_params"),
        "drift": m.wavelet.max_drift(),
        "psnr": res["test_mean"]["PSNR"],
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 74)
    print("A) 默认 bound=0.5 的 PR 臂：fix_arms10_* (n=10) vs fix_bnd05_* (n=3)")
    print("=" * 74)
    for prefix in ("fix_arms10_pr", "fix_bnd05", "new_pr", "new_bnd05", "w3_pr"):
        rs = [r for r in (load(os.path.basename(d))
                          for d in sorted(glob.glob(os.path.join(HERE, "runs", prefix + "*")))) if r]
        if not rs:
            print("  %-16s (无)" % prefix)
            continue
        dr = [r["drift"] for r in rs if not r["old_arch"]]
        ps = [r["psnr"] for r in rs]
        bl = sorted({r.get("bound") for r in rs if not r["old_arch"]})
        print("  %-16s n=%-3d bound=%s   drift %s   PSNR %.4f"
              % (prefix, len(rs), bl,
                 ("%.4f" % np.mean(dr)) if dr else "n/a", np.mean(ps)))
        print("      seeds: %s" % ["%.4f" % x for x in ps])

    print()
    print("=" * 74)
    print("B) E1 容量扫描：按 (n_conv) 分组的 PR 与 fixed，算 Δ")
    print("=" * 74)
    rows = {}
    for d in sorted(glob.glob(os.path.join(HERE, "runs", "*"))):
        tag = os.path.basename(d)
        r = load(tag)
        if not r or r["old_arch"]:
            continue
        if not tag.startswith(("e1_", "fix_arms10_", "new_", "w3_")):
            continue
        kind = ("pr" if "_pr_" in tag else
                "fixed" if "_fixed_" in tag else
                "unc" if "_unconstrained_" in tag else None)
        if kind is None:
            continue
        key = (r["n_conv"], r["n_params"])
        rows.setdefault(key, {}).setdefault(kind, []).append(r["psnr"])

    print("  %-26s %-8s %-22s %-22s %s" % ("(n_conv, n_params)", "kind", "n", "mean PSNR", "Δ(PR-fixed)"))
    for key in sorted(rows, key=lambda k: (k[0], k[1] or 0)):
        g = rows[key]
        pr = g.get("pr", [])
        fx = g.get("fixed", [])
        if pr and fx:
            print("  %-26s PR n=%d mean=%.4f | fixed n=%d mean=%.4f | Δ=%+.4f"
                  % (str(key), len(pr), np.mean(pr), len(fx), np.mean(fx),
                     np.mean(pr) - np.mean(fx)))
        else:
            print("  %-26s %s" % (str(key), {k: len(v) for k, v in g.items()}))

    print()
    print("  注：old_arch=True 的运行（checkpoint 含 dwt.*）无法用当前模型加载，已跳过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
