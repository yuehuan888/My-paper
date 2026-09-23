#!/usr/bin/env bash
# AAPM-Mayo 2016 Low Dose CT Grand Challenge —— 3mm B30 配对子集下载
#
# 来源：Kaggle `abhishekpjiju/training-image-data`（11.27 GB 整包）
#
# 为什么不用整包
# ------------------------------------------------------------------
# 整包 11,269,538,902 B，其中我们只需要 "3mm B30" 那部分。实测该部分在
# ZIP 内**完全连续**（4,756 个条目的 local header 偏移首尾相接，0 个 >1MB 空洞），
# 因此可以只做**一次 1.482 GB 的 Range 请求**，拿到后再在本地顺序解压，
# 省掉 9.8 GB 无用流量（11.27 GB -> 1.48 GB，约 7.6 倍）。
#
#   SPAN_START = 8,147,679,548
#   SPAN_END   = 9,629,380,728
#   SPAN_BYTES = 1,481,701,181   (1.482 GB)
#
# 实测数据规格（已逐个 DICOM tag 核对，非推测）
# ------------------------------------------------------------------
#   CT/<患者>/3mm B30/…  SliceThickness = 3       ✓ 3mm
#                        ConvolutionKernel = B30f ✓ B30
#                        Rows = Columns = 512     ✓ 512×512
#                        PixelSpacing = 0.6640625 ✓
#                        BitsAllocated = 16, Modality = CT
#   10 位患者全覆盖：L067 L096 L109 L143 L192 L286 L291 L310 L333 L506
#   full_3mm 2378 张 + quarter_3mm 2378 张，逐患者两侧数量**完全相等**（真配对）
#
# ⚠️ 网络实测要点（本机代理环境，2026-09-15 实测）
# ------------------------------------------------------------------
# 1. Kaggle 的 /api/v1/datasets/download 端点**无需登录**即 302 到
#    storage.googleapis.com 的预签名 URL。**不要缓存该直链**——每轮重新打
#    Kaggle 端点取新签名。
# 2. storage.googleapis.com 本机**时通时断**：同一条命令可能 0 字节失败，
#    重试即成功。所以重试循环必须包在**外层 shell**，不能指望单条 curl。
# 3. 支持 Range（206 + Content-Range），断点续传成立。
# 4. **不要加大并行度**：实测 4 路并行 120 s 仅 1 路成功、聚合约 68 KiB/s，
#    不优于单流。瓶颈在代理路由，不在连接数。
# 5. 单流实测约 50–120 KiB/s。1.482 GB 预计 **3.5–8 小时**。
#
# 用法：
#   bash download_aapm.sh          # 下载 + 解压（默认全部）
#   bash download_aapm.sh status   # 只看进度
#   bash download_aapm.sh extract  # 只解压（span 已下完时）

set -u

REF="abhishekpjiju/training-image-data"
URL="https://www.kaggle.com/api/v1/datasets/download/${REF}"
SPAN_START=8147679548
SPAN_END=9629380728
SPAN_BYTES=1481701181

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/raw"
SPAN="$DIR/aapm_3mmB30_span.bin"
OUT="$DIR/aapm_mayo_3mm"
LOCK="$DIR/.download.lock"
mkdir -p "$DIR"

# ---------------------------------------------------------------- 单实例锁
# 为什么需要：2026-09-15 实测踩坑 —— 误以为后台任务已死，又起了一个实例，
# 结果**两个进程同时写同一个 span 与 .part**：cat >> 交错、rm 报
# "Device or resource busy"，下载文件被污染，且两边都以为自己在续传。
# 既然续传靠的是"本地已下字节数"，两个写者必然互相破坏。
# 用 mkdir 做原子锁（比 touch 安全：mkdir 在 POSIX 下是原子的）。
if [ "${1:-run}" != "status" ]; then
  if ! mkdir "$LOCK" 2>/dev/null; then
    owner=$(cat "$LOCK/pid" 2>/dev/null || echo "?")
    if [ -n "$owner" ] && kill -0 "$owner" 2>/dev/null; then
      echo "[已在运行] PID $owner 持有锁。不要并行跑——会污染下载。" >&2
      echo "  查看进度： bash $0 status" >&2
      exit 1
    fi
    echo "[清掉陈旧锁] 原 PID $owner 已不存在"
    rm -rf "$LOCK"; mkdir "$LOCK" || exit 1
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT INT TERM
fi

# ⚠️ 两个函数必须分开，不能合并：
#
#   committed() —— **只算已追加进 $SPAN 的字节**。循环用它算续传起点 `from`。
#                  绝不能把 .part 算进来：.part 是"本轮正在下、尚未追加"的数据，
#                  把它计入会让 `from` 跳过一段还没落盘的内容 —— 静默丢数据。
#
#   progress()  —— 只看进度时用，算 span + .part。因为每轮 curl 有 --max-time 900，
#                  只看 $SPAN 的话最多 15 分钟才跳一次，`status` 会一直显示 0%。
committed() { stat -c%s "$SPAN" 2>/dev/null || echo 0; }

progress() {
  local a b
  a=$(committed)
  # 临时分片现在带唯一后缀（$SPAN.part.<轮次>），故用 glob 汇总
  b=$(du -cb "$SPAN".part.* 2>/dev/null | tail -1 | awk '{print $1}')
  echo $((a + ${b:-0}))
}

if [ "${1:-run}" = "status" ]; then
  got=$(progress)
  echo "$got / $SPAN_BYTES bytes  ($(awk "BEGIN{printf \"%.2f\", 100*$got/$SPAN_BYTES}")%)"
  exit 0
fi

# ---------------------------------------------------------------- 解压
do_extract() {
  echo "=== 解压 3mm B30 -> $OUT ==="
  SPAN="$SPAN" OUT="$OUT" python - <<'PYEOF'
import os, struct, zlib, sys
span_path=os.environ['SPAN']; out=os.environ['OUT']
d=open(span_path,'rb').read()
off=0; n=0; written=0
while off+30 <= len(d):
    sig,ver,flg,meth,tm,dt,crc,csize,usize,nlen,elen = struct.unpack('<IHHHHHIIIHH', d[off:off+30])
    if sig != 0x04034b50:
        print(f"  [停止] 偏移 {off} 处不是 local header (sig={sig:#x})"); break
    name=d[off+30:off+30+nlen].decode('utf-8','replace')
    extra=d[off+30+nlen:off+30+nlen+elen]
    # 解析 zip64 extra 取真实 size
    q=0; real_c=csize; real_u=usize
    while q+4<=len(extra):
        hid,hsz=struct.unpack('<HH',extra[q:q+4]); q+=4
        if q+hsz>len(extra): break
        if hid==0x0001:
            z=extra[q:q+hsz]; k=0
            if usize==0xFFFFFFFF and k+8<=len(z): real_u=struct.unpack('<Q',z[k:k+8])[0]; k+=8
            if csize==0xFFFFFFFF and k+8<=len(z): real_c=struct.unpack('<Q',z[k:k+8])[0]; k+=8
            break
        q+=hsz
    data=off+30+nlen+elen
    if meth==8:
        try: raw=zlib.decompress(d[data:data+real_c], -15)
        except Exception as e:
            print(f"  [解压失败] {name}: {e}"); break
    elif meth==0:
        raw=d[data:data+real_c]
    else:
        print(f"  [不支持压缩法 {meth}] {name}"); break
    # 路径重写: Training_Image_Data/3mm B30/<dose>/<dose>/<pid>/<dose>/<file>
    #        -> <dose>/<pid>/<dose>/<file>   （与 prepare_aapm.py 期望一致）
    # 去掉前 3 段（Training_Image_Data / 3mm B30 / 多出的那层 <dose>），
    # 使 src/full_3mm/<pid>/full_3mm 与 src/quarter_3mm/<pid>/quarter_3mm 同时成立。
    parts=name.split('/')
    if len(parts)>=6 and parts[0]=='Training_Image_Data':
        rel='/'.join(parts[3:])
    else:
        rel=name
    dst=os.path.join(out, rel.replace('/', os.sep))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst,'wb') as f: f.write(raw)
    written+=len(raw); n+=1
    off = data+real_c
print(f"  解压 {n} 个文件，{written:,} 字节")
PYEOF
  echo "=== 校验 ==="
  for dose in full_3mm quarter_3mm; do
    tot=$(find "$OUT/$dose" -name '*.IMA' 2>/dev/null | wc -l)
    echo "  $dose: $tot 个 .IMA"
  done
  echo "  患者: $(ls "$OUT/full_3mm" 2>/dev/null | tr '\n' ' ')"
}

if [ "${1:-run}" = "extract" ]; then do_extract; exit $?; fi

# ---------------------------------------------------------------- 下载
echo "=== AAPM-Mayo 3mm B30 <- Kaggle ($REF) ==="
echo "Range $SPAN_START-$SPAN_END  ($SPAN_BYTES 字节, 1.482 GB)"

tries=0; last=0
while :; do
  got=$(committed)          # 续传起点必须用 committed，不能用 progress（会跳过未落盘的 .part）
  if [ "$got" -ge "$SPAN_BYTES" ]; then break; fi
  tries=$((tries+1))
  if [ "$tries" -gt 3000 ]; then echo "[放弃] 重试超 3000 次" >&2; exit 1; fi

  want=$((SPAN_BYTES-got))
  from=$((SPAN_START+got))
  # 每轮重新打 Kaggle 端点拿新签名；显式指定 Range 起点（不能用 -C -，因为
  # 本地偏移需加上 SPAN_START 才是归档内偏移）
  # ⚠️ 超时必须短，这一点是实测调出来的，不是拍脑袋：
  #    本机到 storage.googleapis.com 的连接**极不稳定** —— 同一命令连测三次，
  #    一次 0 字节直接失败、一次 65 KB/s、一次 209 KB/s（另一次独立测试到 522 KB/s）。
  #    失败的连接**是挂起而不是快速报错**，所以 --max-time 越大，单次失败浪费越多。
  #    原先 900s：一次挂起就白等 15 分钟；3000 次重试上限下足以耗掉整晚。
  #    现改为「快失败、快重试」：连接 20s、单次 90s，让重试次数去换吞吐。
  #    每次失败只损失约 90s 而非 900s，且成功的那次照常追加。
  # ⚠️ 每轮用**唯一**临时文件名，这是被一次真实数据损坏换来的教训：
  #    原先复用固定的 "$SPAN.part"，而本机 Windows 会间歇性锁住该文件，
  #    使 `rm -f` 报 "Device or resource busy" 而失败。于是：
  #      旧 .part 残留 -> 本轮 curl 若也失败（未截断它）-> cat 把**上一轮的旧内容
  #      又追加了一遍** -> span 重复累加。实测 span 涨到应有长度的 168%，数据报废。
  #    用唯一名后，"本轮的 .part 只可能由本轮写入"，不存在陈旧内容被重追加的路径。
  seg="$SPAN.part.$tries"
  rm -f "$seg" 2>/dev/null || true
  curl -sL --ssl-revoke-best-effort \
       --connect-timeout 20 --max-time 90 \
       --retry 1 --retry-delay 2 --retry-all-errors \
       -r "${from}-${SPAN_END}" -o "$seg" "$URL" || true

  psz=$(stat -c%s "$seg" 2>/dev/null || echo 0)
  if [ "$psz" -gt 0 ] && [ "$psz" -le "$want" ]; then
    cat "$seg" >> "$SPAN"      # 206 正常分片，追加
  fi
  rm -f "$seg" 2>/dev/null || true

  now=$(committed)
  if [ "$now" -le "$last" ]; then sleep 5; fi   # 无进展则退避
  last=$now
  printf "\r  %d / %d bytes (%.2f%%)  第 %d 轮   " "$now" "$SPAN_BYTES" \
         "$(awk "BEGIN{printf \"%.2f\", 100*$now/$SPAN_BYTES}")" "$tries"
done
echo

got=$(committed)            # 尺寸校验必须用 committed：.part 尚未追加，不能计入
if [ "$got" -ne "$SPAN_BYTES" ]; then
  echo "[尺寸不符] $got / $SPAN_BYTES" >&2; exit 1
fi
echo "[完成] span 下载完毕 ($got 字节)"
do_extract
echo
echo "下一步： python data/prepare_aapm.py --src $OUT --test-patients L506,L067"

# ------------------------------------------------------------------
# 候选源实测记录（2026-09-15，全部 curl 实测，非推测）
# ------------------------------------------------------------------
# ✅ Kaggle abhishekpjiju/training-image-data  <- 本次采用
#      整包 11,269,538,902 B；3mm B30 子集连续段 1,481,701,181 B。
#      10/10 患者；full_3mm 2378 == quarter_3mm 2378；实测 Thickness=3,
#      Kernel=B30f, 512x512, PixelSpacing=0.6640625。
# ✅ Kaggle tridibjyotidas/aapm-ct-1mmb30 （备用，若只需 1mm）
#      3,892,992,234 B；同样 10/10 患者、成对（560+560 / 823+823 / … 共 11,872 个
#      .IMA），实测 Kernel=B30f, 512x512, 但 SliceThickness=1（非 3mm）。
# ❌ Zenodo record 13362750 —— 是"胰腺/肝转移 TCIA"数据（50 例），与 AAPM-Mayo
#      2016 无关（5,794,807,836 B）。Zenodo 全站检索亦无 2016 挑战赛数据。
# ❌ 官方 AAPM Box aapm.app.box.com —— DNS 污染(37.61.54.158 / 2001::1) + 连接
#      挂起，恒为 000；app.box.com 同 000；api.box.com 401（共享链接接口需鉴权）。
# ❌ TCIA —— www./services./wiki.cancerimagingarchive.net 全 000。
#      LDCT-and-Projection-data 仍需人工签署 Restricted License。
# ❌ ModelScope —— 检索 API 正常（用 LoDoPaB 反证过），但 aapm/LDCT/low dose CT/
#      medical CT 查询 TotalCount 均为 0，无任何 AAPM-Mayo 镜像。
# ❌ HuggingFace —— aapm 检索仅 3 条且无关。HajihajihaJimmy/aapm16_radon 含真实
#      L067/L096/… 但为 1mm nii.gz 投影域再分发，非 3mm B30 DICOM 成对集。
#      leechnew/L067 是 MP4 视频，同名误报。
# ❌ Kaggle 其他 —— renlinandrew/ldct-image 与 vuongleminh604/aapm-mayo-clinic 实为
#      TCIA LDCT-and-Projection-data（C###/另一批 L###，**不含** Grand Challenge 10 例）；
#      ishak21/aapm-cst 是 3mm 但仅 full-dose 且缩到 256×256。
# ❌ Google Drive / Dropbox —— drive.google.com 等全部 000（多个 GitHub 仓库的
#      数据链接指向 GDrive，均不可达）。
