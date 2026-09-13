#!/usr/bin/env bash
# LoDoPaB-CT 并行分段下载器（带断点续传）
#
# 为什么需要：本机经代理访问 Zenodo 单流约 1 MB/s，55 GB 需 15 小时；
# 4 路并行可到约 3.1 MB/s。故按范围切块并行下载，再拼接。
#
# 用法：
#   bash download.sh test        # ground_truth_test + observation_test（约 4.6 GB）
#   bash download.sh val         # validation 两个包（约 4.5 GB）
#   bash download.sh train       # train 两个包（约 46 GB）
#   bash download.sh all
#
# 特性：
#   - 每个文件切成 N 段并行下载，段文件保留，中断后重跑自动跳过已完成的段
#   - 拼接到最终文件后才删除段文件
#   - 用 .done 标记避免重复下载

set -u
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890

BASE="https://zenodo.org/records/3384092/files"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw"
SEGS=4          # 并行段数
mkdir -p "$DIR"

# 文件清单：名称 字节数（取自 Zenodo API，用于校验完整性）
declare -A SIZES=(
  ["ground_truth_test.zip"]=1582139537
  ["observation_test.zip"]=3000000000
  ["ground_truth_validation.zip"]=1570000000
  ["observation_validation.zip"]=2940000000
  ["ground_truth_train.zip"]=15940000000
  ["observation_train.zip"]=29940000000
)

download_one() {
  local name="$1" size="${2:-0}"
  local out="$DIR/$name"
  local marker="$out.done"

  if [ -f "$marker" ]; then
    echo "  [跳过] $name 已完成"
    return 0
  fi

  # 若未提供字节数，先用 HEAD 探测
  if [ "$size" -eq 0 ]; then
    size=$(curl -sIL --max-time 60 "$BASE/$name?download=1" \
           | awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{gsub(/\r/,"",v); print v}')
    if [ -z "$size" ] || [ "$size" -eq 0 ]; then
      echo "  [错误] 无法获取 $name 的大小" >&2
      return 1
    fi
  fi

  local seg_size=$(( (size + SEGS - 1) / SEGS ))
  echo "  $name  $(( size / 1048576 )) MB  ->  $SEGS 段"

  local pids=()
  for ((i=0; i<SEGS; i++)); do
    local start=$(( i * seg_size ))
    local end=$(( start + seg_size - 1 ))
    [ "$end" -ge "$size" ] && end=$(( size - 1 ))
    local part="$out.part$i"
    # 已完整下载的段跳过
    if [ -f "$part" ] && [ "$(stat -c%s "$part" 2>/dev/null || echo 0)" -eq $(( end - start + 1 )) ]; then
      continue
    fi
    ( curl -sL --retry 5 --retry-delay 3 --max-time 7200 \
        -r "${start}-${end}" -o "$part" "$BASE/$name?download=1" ) &
    pids+=($!)
  done
  for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done

  # 拼接
  rm -f "$out"
  for ((i=0; i<SEGS; i++)); do
    cat "$out.part$i" >> "$out"
  done

  local got=$(stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$got" -eq "$size" ]; then
    rm -f "$out".part*
    touch "$marker"
    echo "  [完成] $name  $(( got / 1048576 )) MB  (校验通过)"
  else
    echo "  [失败] $name  实际 $got / 期望 $size" >&2
    return 1
  fi
}

run_group() {
  case "$1" in
    test)
      download_one ground_truth_test.zip "${SIZES[ground_truth_test.zip]}"
      download_one observation_test.zip   "${SIZES[observation_test.zip]}"
      ;;
    val)
      download_one ground_truth_validation.zip "${SIZES[ground_truth_validation.zip]}"
      download_one observation_validation.zip   "${SIZES[observation_validation.zip]}"
      ;;
    train)
      download_one ground_truth_train.zip "${SIZES[ground_truth_train.zip]}"
      download_one observation_train.zip  "${SIZES[observation_train.zip]}"
      ;;
    all) run_group val; run_group test; run_group train ;;
    *) echo "用法: $0 {test|val|train|all}" >&2; exit 1 ;;
  esac
}

echo "=== LoDoPaB-CT 下载 -> $DIR ==="
run_group "$1"
echo "=== 组 [$1] 结束 ==="
