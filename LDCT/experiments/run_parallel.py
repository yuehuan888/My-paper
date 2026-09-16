"""并发跑多个训练作业，把 GPU 喂饱。

为什么需要
================================================================
实测（RTX 3050 6GB，2026-09-16）单进程训练时 GPU 严重空转：

    GPU 利用率  22–41%
    显存        1.0–1.4 GB / 6 GB   (16–23%)
    功耗        22–40 W

原因不是数据加载。**两个假设都被实测推翻**：

  1. 以为是 `num_workers=0` 串行加载饿住 GPU
     → 实测 workers=4 **更慢**（177.6s vs 162.9s）。Windows 下 worker>0 走
       spawn，进程启动 + 数据序列化开销超过并行收益，而 h5 读取本来就快。

  2. 以为加大 batch 能提速
     → bs=32 确实快 3.1×（52.9s），但 PSNR 掉 0.33 dB；bs=64 快 4.0×，
       掉 0.59 dB。**加速不是免费的**：batch 变大后每轮更新次数变少，
       Adam 的有效学习率变了，必须重调 LR —— 而那会让结果与论文现有
       46 次运行（全是 bs=8）**不再可比**。

真因是 **kernel launch 开销**：模型只有 2,159 参数，7 个子带头 + 小波变换
全是几十微秒级的小算子，bs=8 / patch=128 每次 kernel 干的活不够填启动开销。
显存又只用了 23% —— 那就**并发跑多个进程**，让 GPU 同时有活干。

实测并发收益（不改任何超参，结果依然可比）：
    4 路并发 → GPU 98–100%、显存 2.9 GB、功耗 52–69 W、**2.1× 吞吐**

用法
================================================================
    # 跑一批 (tag, 参数) 作业，最多 N 路并发
    python experiments/run_parallel.py --jobs jobs.txt --parallel 4

jobs.txt 每行一个作业，字段用空白分隔：
    <tag> [--wavelet pr] [--bound 2.0] [--seed 0] ...
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="作业清单文件")
    ap.add_argument("--parallel", type=int, default=4,
                    help="并发数（实测 4 路可达 GPU 98%%；显存约 0.75 GB/路）")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    jobs = []
    for line in open(args.jobs, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag, rest = parts[0], parts[1:]
        # ⚠️ 完成判据必须用 `DONE`，不能只用 `results.json`。
        #    train.py 先写 results.json（:477）再写 DONE（:490）；若进程恰在两者之间
        #    被杀，只用 results.json 判断会把**没跑完的**当成已完成而跳过。
        #    两个都查，缺任一都视为未完成 -> 重跑。
        rd = os.path.join(HERE, "runs", tag)
        if os.path.exists(os.path.join(rd, "DONE")) and \
                os.path.exists(os.path.join(rd, "results.json")):
            print(f"[跳过] {tag} 已完成")
            continue
        jobs.append((tag, rest))

    if not jobs:
        print("没有待跑作业")
        return 0

    n_par = max(1, args.parallel)
    print(f"=== {len(jobs)} 个作业，{n_par} 路并发 ===")
    print("（显存约 %.1f GB；6GB 卡建议 <=6 路）" % (n_par * 0.75))

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    running = []          # [(proc, tag, t0)]
    done, failed = [], []
    t_start = time.time()

    def launch(tag, extra):
        # 每路写自己的日志 —— 并发时混在一个 stdout 里没法排查
        logdir = os.path.join(HERE, "parallel_logs")
        os.makedirs(logdir, exist_ok=True)
        lf = open(os.path.join(logdir, f"{tag}.log"), "wb")
        cmd = [args.python, "-u", os.path.join(HERE, "train.py"), "--tag", tag] + extra
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT)
        p._logfile = lf          # 结束后关闭
        return p, tag, time.time()

    queue = list(jobs)
    while queue or running:
        while queue and len(running) < n_par:
            tag, extra = queue.pop(0)
            running.append(launch(tag, extra))
            print(f"  [启动] {tag}  (并行 {len(running)}/{n_par})")

        still = []
        for p, tag, t0 in running:
            if p.poll() is None:
                still.append((p, tag, t0))
            else:
                el = time.time() - t0
                lf = getattr(p, "_logfile", None)
                if lf:
                    lf.close()
                if p.returncode == 0:
                    done.append((tag, el))
                    print(f"  [完成] {tag}  {el:.0f}s")
                else:
                    failed.append(tag)
                    print(f"  [失败] {tag}  rc={p.returncode} "
                          f"(见 experiments/parallel_logs/{tag}.log)")
        running = still
        if running and not queue:
            time.sleep(2)
        elif running:
            time.sleep(1)

    total = time.time() - t_start
    print(f"\n=== 全部结束：成功 {len(done)}，失败 {len(failed)}，"
          f"总耗时 {total:.0f}s ===")
    if done:
        # ⚠️ 吞吐要用 **总时间 ÷ 作业数**，不能用各个作业墙钟时间的均值 ——
        #    后者把"在队列里等 GPU"也算进了每个作业的耗时，会得出荒谬的
        #    "0.5× 吞吐"（实测踩过）。总时间才是真正的端到端成本。
        per_job = total / len(done)
        base = 163.0        # 单进程基准（bs=8, 30 轮, 本机实测）
        print(f"    吞吐 {per_job:.0f}s/个  （单进程基准 {base:.0f}s → "
              f"{base/per_job:.2f}× 加速）")
        print(f"    串行跑同样 {len(done)} 个的估算耗时: {base*len(done):.0f}s；"
              f"实际 {total:.0f}s，省 {base*len(done)-total:.0f}s")
    if failed:
        print("    失败:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
