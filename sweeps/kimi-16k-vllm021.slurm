#!/bin/bash -l
# P0-A 第一步：Kimi-Linear-48B 在 **vLLM 0.21** 上跑 16K/c64
# 目的：闸门（sweeps/results-kernel-gate-2026-11-15.md）发现 0.21 自带 causal_conv1d 1.5.4
#       且平台换成 HCUPlatform，因此"换镜像"比"补 fla_core"更便宜。
# 判据：16K/c64 两轮都出数且 mean_ttft_ms > 0（bench.sh 自带有效性校验）
# 卡数：TP=4（92 GB 权重），-c 32（每卡 8 CPU）
# 用法：sbatch sweeps/kimi-16k-vllm021.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -J kimi-16k-021
#SBATCH -o /public/home/luql/work/kimi-16k-021/kimi_%j.out

set -uo pipefail

REPO=/public/home/luql/work/zoo-sync
SRC="$REPO/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm"
WORK=/public/home/luql/work/kimi-16k-021
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID; mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

MODEL_HOST=/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct
IMAGE="image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1"
TAR="$IMGDIR/vllm0.21.0-ubuntu22.04-dtk2604-py3.10_v1.tar.gz"
PORT=30000
# backend 由环境变量控制，便于在同一个作业里换；默认不指定（走平台自选）
BACKEND="${BACKEND:-}"

log() { echo "[kimi021 $(date +%H:%M:%S)] $*"; }
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[kimi021] node=$(hostname) tmp_free=${FREE}G backend=${BACKEND:-<auto>}"
echo "[kimi021] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES | head -1)"

[ -d "$MODEL_HOST" ] || { log "模型目录不存在: $MODEL_HOST"; exit 1; }

# ---------- 镜像 ----------
log "load image vllm021"
gunzip -c "$TAR" | $PD load >"$WORK/load-$JID.log" 2>&1 \
  || { log "FAIL load"; tail -5 "$WORK/load-$JID.log"; exit 1; }

# ---------- 仓库副本（容器内外共用 $WORK/repo） ----------
mkdir -p "$WORK/repo"
rsync -a --exclude bench --exclude logs --exclude official-eval "$SRC"/ "$WORK/repo"/
# 覆盖 env.sh：换成 0.21 镜像
cat > "$WORK/repo/env.sh" <<EOF
export MODEL_ID="moonshotai/Kimi-Linear-48B-A3B-Instruct"
export MODEL_PATH="/models/Kimi"
export HOST_MODEL_PATH="$MODEL_HOST"
export SERVED_NAME="kimi-linear-48b"
export IMAGE="$IMAGE"
export HOST_IMAGE_TAR="$TAR"
export FRAMEWORK="vllm"
export FRAMEWORK_VERSION="0.21.0（vllm0.21.0-ubuntu22.04-dtk2604-py3.10 镜像）"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=$PORT
export TP=4
export EP=1
export PP=1
export MAX_LEN=17408
export MEM_FRAC=0.92
export MAX_NUM_SEQS=64
export LOAD_TIMEOUT=2400
export PATCH_ROOT="\$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')"
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs \$MAX_NUM_SEQS"
EOF
chmod -R a+rwX "$WORK/repo"

# ---------- 起服务 ----------
cleanup() { $PD logs vllm-srv > "$WORK/server-logs-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }
trap cleanup EXIT

log "start vLLM 0.21 server"
$PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODEL_HOST":/models/Kimi:ro \
  -v "$WORK":/work \
  -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && bash env_check.sh > env.log 2>&1; cp env.log /work/env.log
source ./env.sh
exec python3 -m vllm.entrypoints.openai.api_server \
  --model \$MODEL_PATH --served-model-name \$SERVED_NAME \
  --host 127.0.0.1 --port \$PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len \$MAX_LEN --gpu-memory-utilization \$MEM_FRAC \
  --tensor-parallel-size \$TP --max-num-seqs \$MAX_NUM_SEQS \
  ${BACKEND:+--attention-backend $BACKEND}
" > "$WORK/server-$JID.log" 2>&1

OK=0
for t in $(seq 1 300); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; log "server up after $((t*10))s"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || {
    log "container gone；日志 → $WORK/srvlog-$JID.txt"
    $PD logs vllm-srv > "$WORK/srvlog-$JID.txt" 2>&1
    grep -aE "Error|error|assert|Assertion|ValueError|TypeError|KeyError|No module|not supported|out of memory|VMFault|Invalid address|died unexpectedly" "$WORK/srvlog-$JID.txt" | grep -avE "core_client|contextlib|async_llm|otel" | head -25
    exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { log "SERVER_NOT_UP"; exit 1; }

# ---------- 压测：先 16K/c8 作对照，再 16K/c64（重点） ----------
cd "$WORK/repo"
run_bench() {  # $1=ilens $2=concs
  log "bench ilens=$1 concs=$2"
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$SRC/preflight_lib.py":/preflight_lib.py:ro \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e BENCH_ILENS="$1" -e BENCH_CONCS="$2" -e BENCH_ROUNDS="1 2" -e READY_TIMEOUT=900 \
    -e PORT=$PORT -e SERVED_NAME=kimi-linear-48b \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning"
}

# CLI 镜像若未在本地，回退到归档
if ! $PD image exists image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1 2>/dev/null; then
  log "load CLI image from archive"
  gunzip -c "$IMGDIR/mimo-sglang-bw10000_1.1.1.tar.gz" | $PD load >"$WORK/load-cli-$JID.log" 2>&1
fi

run_bench 16384 8
cp -f "$WORK/repo/bench/round2.jsonl" "$WORK/bench-16k-c8-$JID.jsonl" 2>/dev/null
run_bench 16384 64
cp -f "$WORK/repo/bench/round2.jsonl" "$WORK/bench-16k-c64-$JID.jsonl" 2>/dev/null

log "server 是否还活着: $(curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && echo alive || echo dead)"
log "done. 产物：$WORK/repo/bench/ 与 $WORK/server-logs-$JID.txt"
