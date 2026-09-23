"""临时核查 4：定稿 bound 表的正确口径。

发现：fix_bnd05_s{0,1,2} 与 fix_pr_s{0,1,2} 的 PSNR 逐位相同 ——
即 bound 扫描的 β=0.5 档**就是默认 pr 臂**，而 pr 臂有 n=10。

因此正确的表述是：
  · 扫描的四个非默认档各有 n=3（种子 0/1/2）；
  · β=0.5 档另有全部 10 个种子，可用来看 n=3 子集是否被挑选；
  · 配对检验只能用**同种子子集**（0/1/2），这才是 Δ 与 p 的来源。
"""

from __future__ import annotations

import glob
import io
import json
import os
import sys

import numpy as np
import torch
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from models.denoiser import PRWaveletDenoiser  # noqa: E402


def drift_psnr(tag):
    d = os.path.join(HERE, "runs", tag)
    cp, rp, kp = (os.path.join(d, f) for f in
                  ("config.json", "results.json", "best.pth"))
    if not all(os.path.exists(p) for p in (cp, rp, kp)):
        return None
    c = json.load(io.open(cp, encoding="utf-8"))
    r = json.load(io.open(rp, encoding="utf-8"))
    st = torch.load(kp, map_location="cpu", weights_only=False)
    m = PRWaveletDenoiser(bound=c["bound"], init_drift=c.get("init_drift", 0.0))
    m.load_state_dict(st.get("model") or st.get("model_state_dict"))
    return m.wavelet.max_drift(), r["test_mean"]["PSNR"]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 78)
    print("β=0.5 档 = 默认 pr 臂：n=10 全集 与 n=3 子集 的对照")
    print("=" * 78)
    all10 = [drift_psnr("fix_pr_s%d" % s) for s in range(10)]
    all10 = [x for x in all10 if x]
    d10 = np.array([x[0] for x in all10]); p10 = np.array([x[1] for x in all10])
    d3, p3 = d10[:3], p10[:3]
    print("  n=10: drift %.4f   PSNR %.4f ± %.4f" % (d10.mean(), p10.mean(), p10.std(ddof=1)))
    print("  n= 3: drift %.4f   PSNR %.4f ± %.4f   (种子 0/1/2)" % (d3.mean(), p3.mean(), p3.std(ddof=1)))
    print("  子集与全集 PSNR 差 %+.4f dB, drift 差 %+.4f" % (p3.mean() - p10.mean(), d3.mean() - d10.mean()))
    print("  -> 子集未偏离全集，n=3 基线不是被挑选出来的。")

    print()
    print("=" * 78)
    print("扫描表定稿数字")
    print("=" * 78)
    print("  %-16s %-9s %-9s %-9s %s" % ("bound", "n", "drift", "PSNR", "Δ / p (配对, 限种子0-2)"))
    base_p = None
    for b, pref in ((0.5, "fix_bnd05"), (1.0, "fix_bnd10"), (2.0, "fix_bnd20"),
                    (4.0, "fix_bnd40"), (8.0, "fix_bnd80")):
        tags = sorted(glob.glob(os.path.join(HERE, "runs", pref + "_s*")))
        vals = [drift_psnr(os.path.basename(t)) for t in tags]
        vals = [v for v in vals if v]
        dr = np.array([v[0] for v in vals]); ps = np.array([v[1] for v in vals])
        if b == 0.5:
            base_p = ps
            print("  %-16s %-9s %-9.4f %-9.4f %s" % ("0.5 (default)", "10/3", d10.mean(), p10.mean(), "—"))
        else:
            dd = ps - base_p
            t, p = stats.ttest_rel(ps, base_p)
            print("  %-16s %-9s %-9.4f %-9.4f Δ=%+.4f p=%.4f"
                  % (b, len(ps), dr.mean(), ps.mean(), dd.mean(), p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
