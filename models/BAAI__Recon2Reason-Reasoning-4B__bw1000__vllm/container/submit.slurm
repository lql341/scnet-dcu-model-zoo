#!/bin/bash -l
# BW1000 × vLLM × Recon2Reason-Reasoning-4B（T02 视觉多模态；与 __sglang 条目配对做跨引擎对比）
# 双镜像编排：vLLM 镜像起服务 / SGLang 镜像跑官方压测客户端
#   容器 A  vLLM 镜像   起服务（--network=host），并在其中跑 env_check（需要 torch）
#   宿主机              跑 smoke.sh / verify.py（只需 curl + python3 标准库）
#   容器 B  SGLang 镜像 跑官方压测客户端 bench.sh（vLLM 镜像不含 sglang.benchmark.serving）
#
# 用法（在仓库根目录提交）：
#   sbatch container/submit.slurm
#   SKIP_BENCH=1 sbatch --export=ALL container/submit.slurm     # 只验到 smoke，跳过压测
#   BENCH_QUICK=1 sbatch --export=ALL container/submit.slurm    # 压测只跑 4K/并发1/1轮（自检用）
#
# 产物落在宿主机 $WORK/repo（env.log / server.log / smoke.log / bench/）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J recon-vllm

set -uo pipefail

# sbatch 会把脚本拷到 spool 目录，$0 不可用；用 SLURM_SUBMIT_DIR 定位仓库根
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在仓库根目录提交（找不到 $REPO/env.sh）"; exit 1; }
source "$REPO/env.sh"

WORK="${WORK:-/public/home/luql/work/recon-vllm}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G  repo=$REPO  work=$WORK"
if [ "$FREE" -lt 68 ]; then echo "[host] NOT_ENOUGH_SPACE (need 68G for two images), exit 42"; exit 42; fi

for spec in "vLLM|$IMAGE|${HOST_IMAGE_TAR:-}" "CLI|$CLI_IMAGE|${HOST_CLI_TAR:-}"; do
  IFS='|' read -r tag img tar <<<"$spec"
  echo "[host] load $tag image"
  if [ -n "$tar" ] && [ -s "$tar" ]; then $PD load -i "$tar" >/tmp/load-$tag-$JID.log 2>&1; fi
  $PD image exists "$img" 2>/dev/null \
    || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$img" >/tmp/load-$tag-$JID.log 2>&1
  $PD image exists "$img" 2>/dev/null || { echo "[host] $tag image unavailable"; tail -5 /tmp/load-$tag.log; exit 1; }
done

# 仓库拷进 $WORK/repo（容器写产物后留在宿主机）
mkdir -p "$WORK/repo"
rsync -a --exclude bench --exclude official-eval "$REPO"/ "$WORK/repo"/ 2>/dev/null
chmod -R a+rwX "$WORK/repo" 2>/dev/null
# 收容器前先把服务端日志落盘，否则 500/崩溃的根因随容器消失（踩过：只剩 1 行 digest）
cleanup() { $PD logs vllm-srv > "$WORK/vllm-logs-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }
trap cleanup EXIT

# ---------- 容器 A：起服务（前台，detached） ----------
echo "[host] start vLLM server container $(date +%T)"
$PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":/models/M:ro \
  -v "$WORK":/work -v "$REPO":/repo:ro \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && bash env_check.sh > env.log 2>&1 && cp env.log /work/repo/env.log
source ./env.sh
exec python3 -m vllm.entrypoints.openai.api_server \
  --model \$MODEL_PATH --served-model-name \$SERVED_NAME \
  --host 127.0.0.1 --port \$PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len \$MAX_LEN --gpu-memory-utilization \$MEM_FRAC \
  --tensor-parallel-size \$TP --max-num-seqs \$MAX_NUM_SEQS
" > "$WORK/server-$JID.log" 2>&1

OK=0
for t in $(seq 1 240); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "[host] server up after $((t*10))s $(date +%T)"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || {
    echo "[host] container gone；完整日志 → $WORK/srvlog-$JID.txt"
    $PD logs vllm-srv > "$WORK/srvlog-$JID.txt" 2>&1
    grep -aE "Error|error|assert|Assertion|ValueError|TypeError|KeyError|No module|not supported|out of memory|No available|world_size|device" "$WORK/srvlog-$JID.txt" | grep -avE "core_client|contextlib|async_llm|otel" | head -20
    exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "[host] SERVER_NOT_UP"; exit 1; }

# ---------- 宿主机：smoke + verify（无需 GPU/python 依赖） ----------
cd "$WORK/repo"
echo "[host] smoke $(date +%T)"
bash smoke.sh || echo "[host] smoke 有失败项，见 smoke.log"
echo "[host] verify selfcheck + verify"
python3 verify.py --selfcheck || true
python3 verify.py || echo "[host] verify.py 被官方基线文件阻塞，见 MISSING.md"

# ---------- 容器 B：官方压测客户端 ----------
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[host] SKIP_BENCH=1，跳过压测"
else
  echo "[host] bench（CLI 容器）$(date +%T)"
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$REPO":/repo:ro \
    -v "$REPO/preflight_lib.py":/preflight_lib.py:ro \
    -v "$HOST_MODEL_PATH":/models/M:ro \
    -e BENCH_QUICK="${BENCH_QUICK:-0}" \
    "$CLI_IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && source ./env.sh && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"
fi

echo "[host] done $(date +%T)"
echo "[host] 产物目录：$WORK/repo"
ls -la "$WORK/repo"/env.log "$WORK/repo"/smoke.log "$WORK/repo"/bench/summary.md 2>/dev/null
