#!/bin/bash -l
# P0-A 第二发：留在 **vLLM 0.20**，逐配置定位 16K/c64 的 KERNEL VMFault
#
# 为什么不再换镜像：0.21 的 vllm_hcu 加载 Kimi 的 3D conv1d 权重即崩（job 1122012），
# 见 models/...__vllm/README.md §7.3。
#
# 崩溃现场证据（job 1120618）：
#   - Running=29 时崩，正好跨过 PIECEWISE cudagraph 的 24→32 桶
#   - MLA 层走 TRITON_MLA，且 CUDAGraphMode 因它降级为 PIECEWISE
# 因此按信息量排序，一次作业内顺序试这几个配置（每个都要重启服务）：
#   A. --enforce-eager                  ← 直接证伪/证实"PLACEWISE cudagraph 是元凶"
#   B. --max-model-len 17408            ← 缓冲越界假设
#   C. --max-model-len 17408 + eager    ← 只在 A、B 都失败时才跑
#
# 判据：16K/c64 两轮都出数且 mean_ttft_ms > 0。
# 编译缓存挂进 $WORK/cache，重启服务时复用，省掉每轮 ~5 min 的 torch.compile。
# 用法：sbatch sweeps/kimi-16k-020-matrix.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 05:00:00
#SBATCH -J kimi-16k-020
#SBATCH -o /public/home/luql/work/kimi-16k-020/mx_%j.out

set -uo pipefail

REPO=/public/home/luql/work/zoo-sync
SRC="$REPO/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm"
WORK=/public/home/luql/work/kimi-16k-020
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

log() { echo "[mx $(date +%H:%M:%S)] $*"; }
echo "[mx] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"
echo "[mx] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES | head -1)"

[ -d "$MODEL_HOST" ] || { log "模型目录不存在"; exit 1; }

# ---------- 镜像（只先加载 vLLM；CLI 镜像等压测前再加载，省一半磁盘峰值） ----------
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

# ---------- 仓库副本（0.20 口径） ----------
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
export MAX_LEN=20480
export LOAD_TIMEOUT=2400
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs \$MAX_NUM_SEQS"
export PATCH_ROOT="\$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
EOF
chmod -R a+rwX "$WORK/repo"

# bench.sh 用 set -u 且依赖 env.sh 里的这些变量；缺一个就整轮压测静默失败（第一次就踩了）
for v in MAX_LEN MAX_NUM_SEQS MEM_FRAC SERVED_NAME PORT FRAMEWORK MODEL_PATH; do
  grep -q "^export $v=" "$WORK/repo/env.sh" || { log "env.sh 缺 $v，退出"; exit 1; }
done

kill_srv() { $PD logs vllm-srv >> "$WORK/allsrv-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }

# 必须在作业退出前删掉 podman 存储：作业结束后 Slurm 会切断节点访问（pam_slurm_adopt），
# 留在 /tmp 里的镜像解包（单镜像 8~31G）就成了谁也删不掉的垃圾。
# 节点 /tmp 只有几百 G，两个镜像 + 容器层就足以写满，下一发作业会以
# "no space left on device" 失败（本日踩过两次：1122426 探测、1122422 矩阵）。
cleanup_all() {
  kill_srv
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_all EXIT

# ---------- 单配置：起服务 → 16K/c64 → 停服务 ----------
run_config() {  # $1=tag  $2=max_model_len  $3=extra_args
  local tag="$1" mlen="$2" extra="$3"
  log "===== CONFIG $tag (max_model_len=$mlen extra='$extra') ====="
  # bench.sh 的前置校验按 $MAX_LEN 探测，必须与服务端一致，否则 17408 的配置会被
  # 拿去问 20480，测出来的是"服务容不下"而不是这个配置本身
  sed -i "s/^export MAX_LEN=.*/export MAX_LEN=$mlen/" "$WORK/repo/env.sh"
  grep -q "^export MAX_LEN=$mlen$" "$WORK/repo/env.sh" || { log "MAX_LEN 改写失败"; return 1; }
  local srvlog="$WORK/srv-$tag-$JID.txt"
  echo "### CONFIG $tag  max_model_len=$mlen  extra=$extra" >> "$WORK/allsrv-$JID.txt"

  $PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODEL_HOST":/models/Kimi:ro \
    -v "$WORK":/work -v "$WORK/cache":/root/.cache/vllm \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$VIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Kimi --served-model-name kimi-linear-48b \
  --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len $mlen --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 4 --max-num-seqs 64 $extra
" > "$WORK/server-$tag-$JID.log" 2>&1

  local OK=0 t
  for t in $(seq 1 300); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; break; }
    $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || break
    sleep 10
  done
  if [ $OK -ne 1 ]; then
    log "CONFIG $tag: 服务未起来"
    $PD logs vllm-srv > "$srvlog" 2>&1
    grep -aE "Error|error|assert|TypeError|ValueError|VMFault|Invalid address|died unexpectedly" "$srvlog" \
      | grep -avE "core_client|contextlib|async_llm|otel" | head -12
    kill_srv; echo "RESULT $tag SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"; return 1
  fi
  log "CONFIG $tag: 服务已就绪（${t}0s 内）"

  load_cli || { kill_srv; echo "RESULT $tag CLI_LOAD_FAILED" >> "$WORK/matrix-$JID.txt"; return 1; }

  rm -f "$WORK/repo/bench/round2.jsonl"
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$SRC/preflight_lib.py":/preflight_lib.py:ro \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e BENCH_ILENS="16384" -e BENCH_CONCS="64" -e BENCH_ROUNDS="1 2" -e READY_TIMEOUT=900 \
    -e PORT=$PORT -e SERVED_NAME=kimi-linear-48b \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$CIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning" | tail -25

  local alive=dead
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && alive=alive
  if [ -s "$WORK/repo/bench/round2.jsonl" ]; then
    cp -f "$WORK/repo/bench/round2.jsonl" "$WORK/bench-$tag-$JID.jsonl"
    echo "RESULT $tag OK alive=$alive $(python3 -c "
import json,sys
rs=[json.loads(l) for l in open('$WORK/bench-$tag-$JID.jsonl') if l.strip()]
for r in rs:
    print('cfg=%s len=%s c=%s ttft=%.1f thr=%.1f' % ('$tag', r.get('random_input_len'), r.get('max_concurrency'), r.get('mean_ttft_ms',0), r.get('total_throughput',0)))
" 2>/dev/null | tr '\n' ';')" >> "$WORK/matrix-$JID.txt"
    log "CONFIG $tag: 有数据，server=$alive"
  else
    echo "RESULT $tag NO_DATA alive=$alive" >> "$WORK/matrix-$JID.txt"
    log "CONFIG $tag: 无数据，server=$alive"
  fi

  $PD logs vllm-srv > "$srvlog" 2>&1
  if [ "$alive" = "dead" ]; then
    log "CONFIG $tag 崩溃现场："
    grep -aE "VMFault|Invalid address|died unexpectedly|EngineDeadError" "$srvlog" | head -5
  fi
  kill_srv

  # 有数据就停：后面的配置是"更贵的备选"，不必再跑
  [ -s "$WORK/bench-$tag-$JID.jsonl" ] && return 0 || return 1
}

: > "$WORK/matrix-$JID.txt"

# A：证伪 cudagraph 假设（信息量最高）
run_config eagerA 20480 "--enforce-eager" && { log "A 成功，停止"; exit 0; }

# B：缓冲越界假设
run_config len17408 17408 "" && { log "B 成功，停止"; exit 0; }

# C：两者叠加
run_config len17408eager 17408 "--enforce-eager" && { log "C 成功，停止"; exit 0; }

log "三个配置都失败——16K/c64 不是配置问题，需改 kernel 或降档"
cat "$WORK/matrix-$JID.txt"
