"""
推理速度与资源占用的实测。

为什么需要
------------------------------------------------------------------
论文的卖点是"极轻量"（2,159 参数）。但**参数量本身不是效率证据**——
审稿人会问：推理多快？占多少显存？
（尤其若投 Journal of Real-Time Image Processing，这三项是硬门槛。）

按规范测量，避免常见错误：
  - **必须预热**：首几次推理含 CUDA 上下文与 cuDNN 算法选择开销，
    不预热会把 warmup 时间摊进均值（本项目曾因此得到 0.5s 的假均值，
    实际第二张起只要 0.01s）
  - **必须 torch.cuda.synchronize()**：否则 time.time() 测的是内核**派发**
    时间而非执行时间
  - 报告**中位数**而非均值（对离群值稳健）

用法
------------------------------------------------------------------
    python experiments/benchmark_speed.py
    python experiments/benchmark_speed.py --ckpt experiments/runs/w3_pr_s0/best.pth
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
import sys
sys.path.insert(0, ROOT)

from models.denoiser import PRWaveletDenoiser, count_parameters  # noqa: E402


@torch.no_grad()
def measure(model, size=512, warmup=50, iters=200, device="cuda", dtype=torch.float32):
    """返回 (中位时延 ms, 均值 ms, 标准差 ms, P95 ms, 峰值显存 MB)。"""
    model.eval()
    x = torch.rand(1, 1, size, size, device=device, dtype=dtype)

    # 预热：不可省
    for _ in range(warmup):
        _ = model(x)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    times.sort()
    return (st.median(times), sum(times) / len(times), st.pstdev(times),
            times[int(0.95 * len(times)) - 1], peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="可选的 checkpoint 路径")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(HERE, "speed_benchmark.json"))
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("无 CUDA，跳过")
        return
    dev = "cuda"

    print("=" * 84)
    print(f"推理速度实测  |  GPU: {torch.cuda.get_device_name(0)}  |  输入 {args.size}×{args.size}")
    print(f"预热 {args.warmup} 次，计时 {args.iters} 次，CUDA 同步，取中位数")
    print("=" * 84)
    print(f"  {'模型':<22}{'参数量':>12}{'中位时延':>12}{'均值':>10}{'标准差':>10}{'峰值显存':>11}")
    print("  " + "-" * 78)

    rows = []

    # 1. 本文方法
    m = PRWaveletDenoiser(wavelet="pr").to(dev)
    if args.ckpt and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        m.load_state_dict(ck["model_state_dict"])
    n = count_parameters(m)
    med, mean, sd, p95, peak = measure(m, args.size, args.warmup, args.iters, dev)
    print(f"  {'PR-Wavelet (ours)':<22}{n:>12,}{med:>11.2f}ms{mean:>9.2f}ms{sd:>9.3f}{peak:>9.0f}MB")
    rows.append({"model": "PR-Wavelet", "params": n, "median_ms": med,
                 "mean_ms": mean, "std_ms": sd, "p95_ms": p95,
                 "peak_mem_mb": peak, "input": args.size})
    del m
    torch.cuda.empty_cache()

    # 2. RED-CNN 基线
    try:
        from models.redcnn import REDCNN
        m2 = REDCNN().to(dev)
        n2 = count_parameters(m2)
        med2, mean2, sd2, p952, peak2 = measure(m2, args.size, args.warmup,
                                                max(20, args.iters // 4), dev)
        print(f"  {'RED-CNN (baseline)':<22}{n2:>12,}{med2:>11.2f}ms{mean2:>9.2f}ms{sd2:>9.3f}{peak2:>9.0f}MB")
        rows.append({"model": "RED-CNN", "params": n2, "median_ms": med2,
                     "mean_ms": mean2, "std_ms": sd2, "p95_ms": p952,
                     "peak_mem_mb": peak2, "input": args.size})
    except Exception as e:  # noqa: BLE001
        print(f"  RED-CNN: 跳过（{type(e).__name__}: {e}）")

    print("  " + "-" * 78)
    if len(rows) == 2:
        r = rows[1]["params"] / rows[0]["params"]
        s = rows[1]["median_ms"] / rows[0]["median_ms"]
        print(f"  本文方法参数少 {r:.0f} 倍，推理快 {s:.1f} 倍")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "warmup": args.warmup,
                   "iters": args.iters, "results": rows}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
