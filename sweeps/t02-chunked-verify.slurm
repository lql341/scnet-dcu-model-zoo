#!/bin/bash -l
# Recon2Reason-4B / RoboBrain2.5-8B × SGLang：验证 chunked-prefill 4096 的收益
#
# 背景：Kimi × SGLang 实测 chunked-prefill 4096 让 16K/c64 +5.5%、16K/c8 +8.3%
#   （sweeps/results-16k-tuning-2026-11-05.md）。这两个 T02 条目的 16K 档从未扫过该参数
#   （server_args 里 chunked_prefill_size=8192 默认值），值得补一格。
#   预期不确定：它们是视觉模型，但 6 档基准用 random-ids 纯文本，不涉图像预处理，
#   prefill 行为应与 Kimi 同类；且两者单卡 KV cache 上限低（33万/27万 tokens），
#   chunked 对 KV 压力的平滑作用理论上更明显。
#
# 设计：每条目 A/B 两配置，各跑 16K/c64 + 16K/c8（受 chunked 影响最大的两档）。
#   A = 当前交付配置（--attention-backend triton --enable-multimodal）
#   B = A + --chunked-prefill-size 4096
# 判定阈值沿用 Kimi：>2% 有效；±2% 噪声；更差则记录。
#
# 用法：sbatch sweeps/t02-chunked-verify.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -J t02-chunk
#SBATCH -o /public/home/luql/work/t02-chunked/out_%j.log

set -uo pipefail

REPO=/public/home/luql/work/zoo-sync
WORK=/public/home/luql/work/t02-chunked
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID
mkdir -p "$XDG_RUNTIME_DIR" "$WORK/repo" /tmp/emptycerts
chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
ITAR="$IMGDIR/mimo-sglang-bw10000_1.1.1.tar.gz"
PORT=30000
MLEN=32768

log() { echo "[tc $(date +%H:%M:%S)] $*"; }
echo "[tc] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"
echo "[tc] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES 2>/dev/null | head -1)"

FREE_G=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
[ "$FREE_G" -ge 45 ] || { log "NOT_ENOUGH_SPACE: 需要 ≥45G"; exit 42; }

$PD image exists "$IMAGE" 2>/dev/null || {
  log "load image"
  gunzip -c "$ITAR" | $PD load >"$WORK/load-$JID.log" 2>&1 \
    || { log "FAIL load"; tail -5 "$WORK/load-$JID.log"; exit 1; }
}

cleanup_all() {
  $PD logs sgl-srv >> "$WORK/allsrv-$JID.txt" 2>&1
  $PD rm -f sgl-srv >/dev/null 2>&1
  $PD rmi -af >/dev/null 2>&1; $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_all EXIT

kill_srv() { $PD logs sgl-srv >> "$WORK/allsrv-$JID.txt" 2>&1; $PD rm -f sgl-srv >/dev/null 2>&1; }

# $1=模型宿主路径 $2=挂载名 $3=served_name $4=EXTRA
start_srv() {
  local mhost="$1" mname="$2" sname="$3" extra="$4"
  $PD run -d --name sgl-srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$mhost":/models/$mname:ro \
    -v "$WORK":/work \
    -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
    "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export PATH=/opt/dtk/bin:\$PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
exec python3 -m sglang.launch_server \
  --model-path /models/$mname --served-model-name $sname \
  --tp-size 1 --context-length $MLEN \
  --mem-fraction-static 0.85 --max-running-requests 64 \
  --host 127.0.0.1 --port $PORT --trust-remote-code $extra
" > "$WORK/server-$sname-$RUNTAG-$JID.log" 2>&1
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

# $1=条目目录名 $2=ilens $3=conc $4=tag
run_bench() {
  local entry="$1" ilen="$2" conc="$3" tag="$4"
  rm -f "$WORK/repo/bench/"*.jsonl
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$REPO"/models/$entry/preflight_lib.py:/preflight_lib.py:ro \
    -e BENCH_ILENS="$ilen" -e BENCH_CONCS="$conc" -e BENCH_ROUNDS="1" -e READY_TIMEOUT=900 \
    -e PORT=$PORT \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning" | tail -8
  [ -s "$WORK/repo/bench/round1.jsonl" ] && cp -f "$WORK/repo/bench/round1.jsonl" "$WORK/$tag-$JID.jsonl"
}

: > "$WORK/matrix-$JID.txt"

#          条目名                                     宿主模型路径                                        served_name
declare -A ENTRY_HOST=(
  [recon]="BAAI__Recon2Reason-Reasoning-4B__bw1000__sglang:/public/home/luql/models/Recon2Reason-Reasoning-4B:recon2reason-4b:Recon"
  [robo]="BAAI__RoboBrain2.5-8B-NV__bw1000__sglang:/public/home/luql/models/RoboBrain2.5-8B-NV:robobrain2.5-8b-nv:Robo"
)

for key in recon robo; do
  IFS=':' read -r entry mhost sname label <<< "${ENTRY_HOST[$key]}"
  log "########## 条目 $label（$sname） ##########"
  rsync -a --exclude bench --exclude logs --exclude official-eval "$REPO"/models/$entry/ "$WORK/repo"/
  chmod -R a+rwX "$WORK/repo"

  declare -A VARIANTS=(
    [A_base]="--attention-backend triton --trust-remote-code --enable-multimodal"
    [B_chunked4096]="--attention-backend triton --trust-remote-code --enable-multimodal --chunked-prefill-size 4096"
  )
  for name in A_base B_chunked4096; do
    kill_srv
    RUNTAG="${label}_${name}"
    log "--- 变体 $name ---"
    start_srv "$mhost" "$label" "$sname" "${VARIANTS[$name]}"
    if ! wait_srv; then
      log "  $name 服务未起来"
      $PD logs sgl-srv > "$WORK/srv-$RUNTAG-$JID.txt" 2>&1
      grep -aE "Error|assert|ValueError|VMFault|died" "$WORK/srv-$RUNTAG-$JID.txt" | head -8
      echo "RESULT $RUNTAG SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
      continue
    fi
    log "  $name 就绪"
    for pair in "16384 64" "16384 8"; do
      set -- $pair
      log "  跑 ${1}/c${2}"
      run_bench "$entry" "$1" "$2" "${RUNTAG}_${1}_c${2}"
    done
    python3 - "$WORK" "$RUNTAG" "$JID" >> "$WORK/matrix-$JID.txt" <<'PY'
import json, sys, glob
work, name, jid = sys.argv[1], sys.argv[2], sys.argv[3]
for p in sorted(glob.glob(f"{work}/{name}_*_c*_{jid}.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        d = json.loads(l)
        if d["random_input_len"] != 16384: continue
        print("RESULT %s c%d ok=%d out=%.2f ttft=%.0f tpot=%.1f"
              % (name, d["max_concurrency"], d["completed"],
                 d["output_throughput"], d["mean_ttft_ms"], d["mean_tpot_ms"]))
PY
  done
done

echo "==================== 汇总 ===================="
cat "$WORK/matrix-$JID.txt"
python3 - "$WORK" "$JID" <<'PY'
import re, sys
work, jid = sys.argv[1], sys.argv[2]
vals = {}
for line in open(f"{work}/matrix-{jid}.txt"):
    m = re.match(r"RESULT (\S+?)_(A_base|B_chunked4096) c(\d+).*out=([\d.]+)", line)
    if m: vals[(m.group(1), m.group(2), int(m.group(3)))] = float(m.group(4))
for label in ("Recon", "Robo"):
    for c in (64, 8):
        a = vals.get((label, "A_base", c)); b = vals.get((label, "B_chunked4096", c))
        if a and b:
            print(f"[tc] {label} 16K/c{c}: {a:.2f} → {b:.2f} ({(b/a-1)*100:+.1f}%)")
            if b > a * 1.02:   print(f"[tc]   判定：chunked **有效**（>2%），应进 env.sh")
            elif abs(b/a-1) <= 0.02: print(f"[tc]   判定：±2% 内，视为噪声，不改交付配置")
            else: print(f"[tc]   判定：B 更差，不采用")
        else:
            print(f"[tc] {label} 16K/c{c}: 数据缺失（A={a} B={b}）")
PY
log "产物在 $WORK"
