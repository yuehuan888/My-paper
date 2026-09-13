#!/usr/bin/env bash
# LoDoPaB-CT 从 ModelScope 镜像下载（比 Zenodo 快约 1.6 倍）
#
# 为什么换源（2026-09-13 实测）
# ------------------------------------------------------------------
# Zenodo 中位 0.738 MB/s，ModelScope 中位 1.203 MB/s（同窗口对照，1.63 倍）。
# 全量 55.3 GB：ModelScope 约 12.8 h vs Zenodo 约 20.8 h，省约 8 小时。
#
# 完整性已验证（非推测）：
#   - Content-Length 与 Zenodo 官方逐位相同（observation_train = 29,938,738,627）
#   - 偏移 1e9 处各抓 1 MiB，md5 一致（3e7da5afe7ffa9106635cfb875a7a4cb）
#   - 5 个偏移的 Range 请求全部返回 206，支持断点续传
#
# ⚠️ 关键实现细节
# ------------------------------------------------------------------
# CDN 直链的 auth_key 只有**分钟级**有效期。因此：
#   1. 必须走 /repo 端点**现取现用**，不要缓存 302 之后拿到的直链
#   2. `-C -`（断点续传）必须放在**外层 shell 循环**里，而不是单条长 curl——
#      长连接中途会因签名过期断掉，每轮重新解析 /repo 可拿到新 key 并从断点继续
#   3. 需重试逻辑：实测偶发 HTTP 400（size=237），重试即成功，不能裸跑
#
# 实测不建议做的事
# ------------------------------------------------------------------
#   - 不要加大并行度：16 路并行 Zenodo 仅 2.09 MB/s，16 路未显著优于 4 路，
#     瓶颈在代理路由不在段数
#   - 不要直连：ModelScope 直连 1.02 MB/s < 走代理 1.20 MB/s（与"国内 CDN 该直连"
#     的直觉相反）。Zenodo 直连更是只有 0.145 MB/s。
#   - 不要把段切碎：每条新连接付 TLS+TTFB 约 4.2 s，16 MiB 段按 0.8 MB/s
#     传 20 s 就要付 12% 建连开销
#
# 用法：
#   bash download_modelscope.sh              # 全部
#   bash download_modelscope.sh small        # 只下 test + validation（约 9.1 GB）

set -u
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

BASE='https://modelscope.cn/api/v1/datasets/OmniData/LoDoPaB-CT/repo?Revision=master&FilePath='
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw"
mkdir -p "$DIR"

declare -A SIZES=(
  ["ground_truth_test.zip"]=1582139537
  ["observation_test.zip"]=2996574366
  ["ground_truth_validation.zip"]=1567086812
  ["observation_validation.zip"]=2944573582
  ["ground_truth_train.zip"]=15940074239
  ["observation_train.zip"]=29938738627
)

download_one() {
  local f="$1" expect="$2"
  local out="$DIR/$f"
  local marker="$out.done"

  if [ -f "$marker" ]; then
    echo "  [跳过] $f 已完成"
    return 0
  fi

  echo "  $f  期望 $(( expect / 1048576 )) MB"
  local tries=0
  while :; do
    local got=0
    [ -f "$out" ] && got=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$got" -ge "$expect" ]; then break; fi
    tries=$((tries+1))
    if [ "$tries" -gt 200 ]; then
      echo "  [放弃] $f 重试超过 200 次" >&2
      return 1
    fi
    # 每轮重新解析 /repo 拿新的 auth_key，并从断点续传
    curl -L -C - --max-time 7200 --retry 5 --retry-delay 5 --retry-all-errors \
         -s -o "$out" "${BASE}raw%2F${f}.zip" || true
    local now=0
    [ -f "$out" ] && now=$(stat -c%s "$out" 2>/dev/null || echo 0)
    if [ "$now" -le "$got" ]; then
      sleep 10                       # 无进展则退避，避免死循环空转
    fi
    printf "\r    %s  %6.1f / %6.1f MB (%.1f%%)  第 %d 轮" \
           "$f" "$(echo "$now/1048576" | bc -l)" "$(echo "$expect/1048576" | bc -l)" \
           "$(echo "100*$now/$expect" | bc -l)" "$tries"
  done
  echo
  local got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" -eq "$expect" ]; then
    touch "$marker"
    echo "  [完成] $f  校验通过"
  else
    echo "  [尺寸不符] $f  $got / $expect" >&2
    return 1
  fi
}

echo "=== LoDoPaB-CT @ ModelScope -> $DIR ==="
if [ "${1:-all}" = "small" ]; then
  download_one ground_truth_test.zip       "${SIZES[ground_truth_test.zip]}"
  download_one observation_test.zip        "${SIZES[observation_test.zip]}"
  download_one ground_truth_validation.zip "${SIZES[ground_truth_validation.zip]}"
  download_one observation_validation.zip  "${SIZES[observation_validation.zip]}"
else
  for f in ground_truth_test.zip observation_test.zip \
           ground_truth_validation.zip observation_validation.zip \
           ground_truth_train.zip observation_train.zip; do
    download_one "$f" "${SIZES[$f]}"
  done
fi
echo "=== MS 下载结束 ==="
