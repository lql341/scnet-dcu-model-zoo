#!/bin/bash -l
# BW1000 × SGLang × RoboBrain2.5-8B-NV 一键复现（T02 视觉多模态）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J robo-accept
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/luql/work/robob-sglang}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

# 磁盘守门：/tmp 满会让 mkdir 直接失败（实测踩过）
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"

if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then
  $PD load -i "$HOST_IMAGE_TAR" >/tmp/load-$JID.log 2>&1
fi
$PD image exists "$IMAGE" 2>/dev/null \
  || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/load-$JID.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/load-$JID.log; exit 1; }

mkdir -p "$WORK/repo" && rsync -a --exclude bench --exclude logs "$REPO"/ "$WORK/repo"/ 2>/dev/null
echo "[host] start $(date +%T)"
$PD run -d --name robo-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work \
  -v "$REPO/preflight_lib.py":/preflight_lib.py:ro \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_ROCM_FUSED_DECODE_MLA=0
cd /work/repo && bash env_check.sh; nohup bash run.sh > /dev/null 2>&1 & exec tail -f /dev/null" > /dev/null 2>&1

OK=0
for t in $(seq 1 180); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "[host] server up $((t*10))s $(date +%T)"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q robo-srv || { echo "[host] CONTAINER_GONE"; exit 1; }
  # 容器用 tail -f 保活，因此还要单独查服务进程是否还在
  $PD exec robo-srv bash -lc 'pgrep -f launch_server >/dev/null' 2>/dev/null \
    || { echo "[host] SERVER_PROCESS_DIED at $((t*10))s"; grep -iE "error|NameError|Traceback|assert" "$WORK/repo/logs/server.log" 2>/dev/null | tail -8; exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "[host] SERVER_NOT_UP"; $PD logs robo-srv 2>&1|grep -iE "error|assert|Traceback"|tail -15; exit 1; }

echo "[host] smoke $(date +%T)"
$PD exec robo-srv bash -lc "cd /work/repo && bash smoke.sh" || echo "[host] smoke 有失败项"

# 6 档基准（服务仍在运行）：输入 4K/16K × 输出 1K × 并发 1/8/64，每档 2 轮取第 2 轮
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[host] SKIP_BENCH=1，跳过压测"
else
  echo "[host] bench $(date +%T)"
  $PD exec robo-srv bash -lc "cd /work/repo && bash bench.sh" 2>&1 | grep -viE "subuid|rootless|level=warning" | tail -30
fi
$PD rm -f robo-srv >/dev/null 2>&1
echo "[host] DONE $(date +%T)  产物 $WORK/repo/logs"
