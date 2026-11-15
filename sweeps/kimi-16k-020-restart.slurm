#!/bin/bash -l
# P0-A 判定实验：崩溃究竟是"累计状态"还是"每轮独立的概率事件"
#
# 已有的三组数据（作业 1123145 / 1123874）：
#   A 默认            round1 ✅128/128   round2 ❌ 崩在 Running=29
#   A --enforce-eager round1 ✅128/128   round2 ❌ 崩在 Running=62
#   D --max-num-seqs 24 round1 ✅128/128 round2 ❌ 崩在 Running=23（已成功 38 个请求）
#
# 共性：**round1 永远过，round2 永远崩**。且崩溃点 29/62/23 与并发上限无关。
# 两个竞争假设：
#   H1 累计状态——服务进程跑过一整轮 16K 后，内部某处状态（KDA recurrent state /
#      集合通信 workspace / 显存碎片）被污染，此后必崩。
#   H2 概率事件——每个 16K 长跑轮次有固定概率崩，与服务是否"跑过一轮"无关；
#      round1 只是运气好（样本只有 2 个）。
#
# 本实验：**每轮都重启服务**，各跑一整轮 16K/c64。
#   - 3~4 轮全部通过  → H1 成立。交付口径：每轮之间重启服务即可拿到多轮有效数据；
#                        根因指向进程内状态累积，下一步查 KDA state 与 workspace 释放。
#   - 出现"fresh server 第一轮就崩" → H2 成立。RocmAiter/MLA 的 kernel 在 gfx936 上
#                        本身就是不稳定的，与轮次无关，需要换 backend 或降上下文。
#
# 每轮的输出都单独存成 bench-S<i>-$JID.jsonl，并记录崩溃时的最后一个
# "Running"/"Successful requests"，便于比较"崩在第几个请求 / 第几秒"。
#
# 用法：sbatch sweeps/kimi-16k-020-restart.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -J kimi-restart
#SBATCH -o /public/home/luql/work/kimi-16k-020/rs_%j.out

set -uo pipefail

REPO=/public/home/luql/work/zoo-sync
SRC="$REPO/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm"
WORK=/public/home/luql/work/kimi-16k-020-restart
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID
mkdir -p "$XDG_RUNTIME_DIR" "$WORK/cache" "$WORK/repo" /tmp/emptycerts
chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

MODEL_HOST=/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct
VIMAGE="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
VTAR="$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"
CIMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
CTAR="$IMGDIR/mimo-sglang-bw10000_1.1.1.tar.gz"
PORT=30000
MLEN=20480
NROUND=${NROUND:-4}

log() { echo "[rs $(date +%H:%M:%S)] $*"; }
echo "[rs] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"
echo "[rs] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES | head -1)"

[ -d "$MODEL_HOST" ] || { log "模型目录不存在"; exit 1; }

FREE_G=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
log "tmp_free=${FREE_G}G"
[ "$FREE_G" -ge 45 ] || { log "NOT_ENOUGH_SPACE: 需要 ≥45G，实际 ${FREE_G}G"; exit 42; }

$PD image exists "$VIMAGE" 2>/dev/null || {
  log "load vllm020 image"
  gunzip -c "$VTAR" | $PD load >"$WORK/load-vllm020-$JID.log" 2>&1 \
    || { log "FAIL load vllm020"; tail -5 "$WORK/load-vllm020-$JID.log"; exit 1; }
}

load_cli() {
  $PD image exists "$CIMAGE" 2>/dev/null && return 0
  log "load cli image"
  gunzip -c "$CTAR" | $PD load >"$WORK/load-cli-$JID.log" 2>&1 \
    || { log "FAIL load cli"; tail -5 "$WORK/load-cli-$JID.log"; return 1; }
}

rsync -a --exclude bench --exclude logs --exclude official-eval "$SRC"/ "$WORK/repo"/
cat > "$WORK/repo/env.sh" <<EOF
export MODEL_ID="moonshotai/Kimi-Linear-48B-A3B-Instruct"
export MODEL_PATH="/models/Kimi"
export HOST_MODEL_PATH="$MODEL_HOST"
export SERVED_NAME="kimi-linear-48b"
export IMAGE="$VIMAGE"
export HOST_IMAGE_TAR="$VTAR"
export FRAMEWORK="vllm"
export FRAMEWORK_VERSION="0.20.0"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=$PORT
export TP=4
export EP=1
export PP=1
export MEM_FRAC=0.92
export MAX_NUM_SEQS=64
export MAX_LEN=$MLEN
export LOAD_TIMEOUT=2400
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs \$MAX_NUM_SEQS"
export PATCH_ROOT="\$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
EOF
chmod -R a+rwX "$WORK/repo"

for v in MAX_LEN MAX_NUM_SEQS MEM_FRAC SERVED_NAME PORT FRAMEWORK MODEL_PATH; do
  grep -q "^export $v=" "$WORK/repo/env.sh" || { log "env.sh 缺 $v，退出"; exit 1; }
done

kill_srv() { $PD logs vllm-srv >> "$WORK/allsrv-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }
cleanup_all() {
  kill_srv
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_all EXIT

start_srv() {  # $1=suffix for log names
  local sfx="$1"
  $PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODEL_HOST":/models/Kimi:ro \
    -v "$WORK":/work -v "$WORK/cache":/root/.cache/vllm \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$VIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export PATH=/opt/dtk/bin:\$PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Kimi --served-model-name kimi-linear-48b \
  --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len $MLEN --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 4 --max-num-seqs 64
" > "$WORK/server-$sfx-$JID.log" 2>&1
}

wait_srv() {
  local t
  for t in $(seq 1 300); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && return 0
    $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || return 1
    sleep 10
  done
  return 1
}

srv_alive() { curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; }

# $1 = bench 客户端日志（有 Successful requests），$2 = 服务端日志（有 Running:）
round_stats() {
  local bf="$1" sf="$2" ok last
  ok=$(grep -aE "^Successful requests:" "$bf" 2>/dev/null | tail -1 | awk '{print $3}')
  last=$(grep -aE "Running:" "$sf" 2>/dev/null | tail -1 | sed 's/.*\] //')
  echo "ok=${ok:-?} last=[${last:-无}]"
}

: > "$WORK/matrix-$JID.txt"
FAILS=0

for i in $(seq 1 $NROUND); do
  log "=========== ROUND $i / $NROUND （全新服务进程） ==========="
  start_srv "r$i"
  if ! wait_srv; then
    log "ROUND $i: 服务未起来"
    $PD logs vllm-srv > "$WORK/srv-r$i-$JID.txt" 2>&1
    grep -aE "Error|assert|TypeError|ValueError|VMFault|Invalid address|died unexpectedly" "$WORK/srv-r$i-$JID.txt" \
      | grep -avE "core_client|contextlib|async_llm|otel" | head -12
    echo "RESULT R$i SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
    kill_srv; FAILS=$((FAILS+1)); [ $FAILS -ge 2 ] && { log "连续 2 次失败，停止"; break; }
    continue
  fi
  log "ROUND $i: 服务已就绪"
  load_cli || { kill_srv; echo "RESULT R$i CLI_LOAD_FAILED" >> "$WORK/matrix-$JID.txt"; FAILS=$((FAILS+1)); continue; }

  rm -f "$WORK/repo/bench/"*.jsonl "$WORK/repo/bench/round1_16384_1024_c64.log"
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$SRC/preflight_lib.py":/preflight_lib.py:ro \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e BENCH_ILENS="16384" -e BENCH_CONCS="64" -e BENCH_ROUNDS="1" -e READY_TIMEOUT=900 \
    -e PORT=$PORT -e SERVED_NAME=kimi-linear-48b \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$CIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning" | tail -20

  alive=dead; st=""
  srv_alive && alive=alive
  # 服务端日志必须先抓，round_stats 的 Running/崩溃点都在里面
  $PD logs vllm-srv > "$WORK/srv-r$i-$JID.txt" 2>&1
  st=$(round_stats "$WORK/repo/bench/round1_16384_1024_c64.log" "$WORK/srv-r$i-$JID.txt")

  if [ -s "$WORK/repo/bench/round1.jsonl" ]; then
    cp -f "$WORK/repo/bench/round1.jsonl" "$WORK/bench-S$i-$JID.jsonl"
    line=$(python3 -c "
import json
rs=[json.loads(l) for l in open('$WORK/bench-S$i-$JID.jsonl') if l.strip()]
r=rs[-1]
print('ttft=%.1f tpot=%.1f thr=%.2f dur=%.1f' % (r.get('mean_ttft_ms',0), r.get('mean_tpot_ms',0), r.get('total_throughput',0), r.get('duration',0)))
" 2>/dev/null)
    echo "RESULT R$i PASS alive=$alive $line $st" >> "$WORK/matrix-$JID.txt"
    log "ROUND $i: ✅ $line $st"
    FAILS=0
  else
    echo "RESULT R$i NO_DATA alive=$alive $st" >> "$WORK/matrix-$JID.txt"
    log "ROUND $i: ❌ 无数据 alive=$alive $st"
    FAILS=$((FAILS+1))
  fi

  if [ "$alive" = "dead" ]; then
    log "ROUND $i 崩溃现场："
    grep -aE "VMFault|Invalid address|died unexpectedly|EngineDeadError" "$WORK/srv-r$i-$JID.txt" | head -5
  fi
  kill_srv

  [ $FAILS -ge 2 ] && { log "连续 2 次失败，停止"; break; }
done

log "=========== 汇总 ==========="
cat "$WORK/matrix-$JID.txt"
P=$(grep -c "^RESULT R.* PASS" "$WORK/matrix-$JID.txt" || true)
N=$(grep -c "^RESULT R" "$WORK/matrix-$JID.txt" || true)
R1=$(grep -c "^RESULT R1 PASS" "$WORK/matrix-$JID.txt" || true)
log "通过 $P / $N 轮（首轮 $([ "$R1" = 1 ] && echo 通过 || echo 失败)）"
if [ "$P" = "$N" ] && [ "$N" -ge 3 ]; then
  log "判定：**H1 累计状态成立** —— 全新服务进程可反复成功，崩溃源于进程内状态累积。交付口径=每轮重启服务。"
elif [ "$R1" = 1 ]; then
  log "判定：**部分成立** —— 首轮通过但后续轮失败。H1 弱化为「进程跑满一整轮后成功率下降」，"
  log "        不是「跑过一轮必崩」。需按 RESULT 行的通过率判断，并对照各轮崩溃点。"
else
  log "判定：**H2 概率事件成立** —— 全新服务进程首轮就崩，与轮次无关。需换 backend 或降上下文。"
fi
