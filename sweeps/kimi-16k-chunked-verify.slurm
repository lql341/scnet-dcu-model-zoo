#!/bin/bash -l
# Kimi × SGLang：验证 chunked-prefill 4096 在 16K 档的实际收益
#
# 依据：sweeps/results-16k-tuning-2026-11-05.md 实测（1.4T checkpoint 时代的 5 变体对照）
#   K0 base（graph 开）        16K/c64 232.03   16K/c8 127.71
#   K3 chunked-prefill 4096    16K/c64 244.73（+5.5%）  16K/c8 138.31（+8.3%）
# 但当前交付配置 env.sh 的 EXTRA_ARGS 只有 --attention-backend triton，
# **这处已验证的收益从未进正式配置**（2026-11-15 发现）。
#
# 本作业：同一节点、同一镜像、同一模型，A/B 两配置各跑 1 轮 16K/c64 + 16K/c8
#   A  当前交付配置（无 chunked）
#   B  + --chunked-prefill-size 4096
# 目标：确认 +5% 在当前环境仍成立；成立则 B 进 env.sh，并把 meta.throughput 更新为 B 的实测值。
#
# 为什么只跑 16K：chunked-prefill 只影响 prefill/decode 交错，4K 档 prefill 占比小，
#   且 Kimi 的 4K 档已有稳定数据，不必翻新旧结论。两档 × 2 配置 ≈ 2×(460+83) s + 2 次冷启动。
#
# 用法：sbatch sweeps/kimi-16k-chunked-verify.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -J kimi-chunk
#SBATCH -o /public/home/luql/work/kimi-16k-chunked/out_%j.log

set -uo pipefail

REPO=/public/home/luql/work/zoo-sync
WORK=/public/home/luql/work/kimi-16k-chunked
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID
mkdir -p "$XDG_RUNTIME_DIR" "$WORK/cache" "$WORK/repo" /tmp/emptycerts
chmod 700 "$XDG_RUNTIME_DIR"
# podman 存储放【共享目录】，不要放节点本地 /tmp：
# 镜像解压后 20G+，每个作业一份会撑爆节点 /（2026-11-15 实测把登录节点灌到 99%，
# 另有机次报 no space left on device）。共享盘有 400G+ 余量，且作业结束自动清理。
PSTORE=/public/home/luql/work/podman-$JID
PD="podman --root $PSTORE/root --runroot $PSTORE/runroot --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

MODEL_HOST=/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct
VIMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
VTAR="$IMGDIR/mimo-sglang-bw10000_1.1.1.tar.gz"
PORT=30000
MLEN=20480

log() { echo "[kc $(date +%H:%M:%S)] $*"; }
echo "[kc] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"
echo "[kc] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES 2>/dev/null | head -1)"

[ -d "$MODEL_HOST" ] || { log "模型目录不存在"; exit 1; }
FREE_G=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
[ "$FREE_G" -ge 45 ] || { log "NOT_ENOUGH_SPACE: 需要 ≥45G，实际 ${FREE_G}G"; exit 42; }

$PD image exists "$VIMAGE" 2>/dev/null || {
  log "load image"
  gunzip -c "$VTAR" | $PD load >"$WORK/load-$JID.log" 2>&1 \
    || { log "FAIL load image"; tail -5 "$WORK/load-$JID.log"; exit 1; }
}

rsync -a --exclude bench --exclude logs --exclude official-eval "$REPO"/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/ "$WORK/repo"/
chmod -R a+rwX "$WORK/repo"

cleanup_all() {
  $PD logs sgl-srv >> "$WORK/allsrv-$JID.txt" 2>&1
  $PD rm -f sgl-srv >/dev/null 2>&1
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  $PD rmi -af >/dev/null 2>&1; $PD system prune -af >/dev/null 2>&1
  rm -rf "$PSTORE" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_all EXIT

kill_srv() { $PD logs sgl-srv >> "$WORK/allsrv-$JID.txt" 2>&1; $PD rm -f sgl-srv >/dev/null 2>&1; }

# $1=变体名  $2=EXTRA_ARGS（空格分隔）
start_srv() {
  local name="$1" extra="$2"
  log "启动变体 $name：EXTRA=$extra"
  $PD run -d --name sgl-srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODEL_HOST":/models/Kimi:ro \
    -v "$WORK":/work \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$VIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export PATH=/opt/dtk/bin:\$PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo
exec python3 -m sglang.launch_server \
  --model-path /models/Kimi --served-model-name kimi-linear-48b \
  --tp-size 4 --context-length $MLEN \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --host 127.0.0.1 --port $PORT --trust-remote-code $extra
" > "$WORK/server-$name-$JID.log" 2>&1
}

wait_srv() {
  local t
  for t in $(seq 1 300); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    $PD ps --format '{{.Names}}' 2>/dev/null | grep -q sgl-srv || return 1
    sleep 10
  done
  return 1
}

# 跑一轮指定档位；$1=ilens $2=conc $3=tag
run_bench() {
  local ilen="$1" conc="$2" tag="$3"
  rm -f "$WORK/repo/bench/"*.jsonl
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$REPO"/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/preflight_lib.py:/preflight_lib.py:ro \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e BENCH_ILENS="$ilen" -e BENCH_CONCS="$conc" -e BENCH_ROUNDS="1" -e READY_TIMEOUT=900 \
    -e PORT=$PORT -e SERVED_NAME=kimi-linear-48b \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$VIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning" | tail -12
  [ -s "$WORK/repo/bench/round1.jsonl" ] && cp -f "$WORK/repo/bench/round1.jsonl" "$WORK/$tag-$JID.jsonl"
}

: > "$WORK/matrix-$JID.txt"

declare -A VARIANTS=(
  [A_current]="--attention-backend triton"
  [B_chunked4096]="--attention-backend triton --chunked-prefill-size 4096"
)

# SKIP_A=1 时只跑 B（A 的数据已在被取消的 1125845 里留存）
VLIST="A_current B_chunked4096"
[ "${SKIP_A:-0}" = "1" ] && VLIST="B_chunked4096"

for name in $VLIST; do
  kill_srv
  log "==================== 变体 $name ===================="
  start_srv "$name" "${VARIANTS[$name]}"
  if ! wait_srv; then
    log "变体 $name 服务未起来"
    $PD logs sgl-srv > "$WORK/srv-$name-$JID.txt" 2>&1
    grep -aE "Error|assert|ValueError|VMFault|died" "$WORK/srv-$name-$JID.txt" | head -8
    echo "RESULT $name SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
    continue
  fi
  log "变体 $name 就绪"

  for pair in "16384 64" "16384 8"; do
    set -- $pair
    log "  跑 ${1}/c${2}"
    run_bench "$1" "$2" "${name}_${1}_c${2}"
    $PD logs sgl-srv > "$WORK/srv-$name-${1}c${2}-$JID.txt" 2>&1
  done

  # 汇总该变体的两个档位
  python3 - "$WORK" "$name" "$JID" >> "$WORK/matrix-$JID.txt" <<'PY'
import json, sys, glob, os
work, name, jid = sys.argv[1], sys.argv[2], sys.argv[3]
for p in sorted(glob.glob(f"{work}/{name}_*_c*_{jid}.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        d = json.loads(l)
        if d["random_input_len"] != 16384: continue
        print("RESULT %s in=%d c%d ok=%d out=%.2f ttft=%.0f tpot=%.1f"
              % (name, d["random_input_len"], d["max_concurrency"], d["completed"],
                 d["output_throughput"], d["mean_ttft_ms"], d["mean_tpot_ms"]))
PY
done

echo "==================== 汇总 ===================="
cat "$WORK/matrix-$JID.txt"

python3 - "$WORK" "$JID" <<'PY'
import re, sys
work, jid = sys.argv[1], sys.argv[2]
vals = {}
for line in open(f"{work}/matrix-{jid}.txt"):
    m = re.match(r"RESULT (\S+) in=\d+ c(\d+) .*out=([\d.]+)", line)
    if m: vals[(m.group(1), int(m.group(2)))] = float(m.group(3))
a64, b64 = vals.get(("A_current", 64)), vals.get(("B_chunked4096", 64))
a8, b8 = vals.get(("A_current", 8)), vals.get(("B_chunked4096", 8))
if a64 and b64:
    print(f"[kc] 16K/c64:  {a64:.2f} → {b64:.2f}  ({(b64/a64-1)*100:+.1f}%)")
if a8 and b8:
    print(f"[kc] 16K/c8 :  {a8:.2f} → {b8:.2f}  ({(b8/a8-1)*100:+.1f}%)")
if a64 and b64:
    if b64 > a64 * 1.02:
        print("[kc] 判定：chunked-prefill 4096 **有效**（>2%），应进 env.sh 并更新 meta.throughput")
    elif abs(b64 / a64 - 1) <= 0.02:
        print("[kc] 判定：差异在 ±2% 内，视为**噪声** —— 保留原因记录，不改交付配置")
    else:
        print("[kc] 判定：B 更差，**不要采用**，并更新 README 说明旧结论已失效")
else:
    print("[kc] 判定：数据不足，需看 server 日志")
PY
log "产物在 $WORK"
