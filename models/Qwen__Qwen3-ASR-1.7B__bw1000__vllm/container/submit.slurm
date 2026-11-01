#!/bin/bash -l
# BW1000 × vLLM × Qwen3-ASR-1.7B 一键复现
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -J asr-accept

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/asr}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 35 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull.log 2>&1 \
  || { tail -3 /tmp/pull.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null

$PD run -d --name asr-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":$MODEL_PATH:ro \
  -v "$WORK":/work "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && bash env_check.sh; nohup bash run.sh > /dev/null 2>&1 & exec tail -f /dev/null" > /dev/null 2>&1

OK=0
for t in $(seq 1 120); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "[host] server up $((t*10))s"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q asr-srv || { echo "[host] CONTAINER_GONE"; $PD logs asr-srv 2>&1|tail -20; exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "[host] SERVER_NOT_UP"; exit 1; }

echo "[host] smoke (容器内执行，音频在 /work/zoo-assets)"
$PD exec asr-srv bash -lc "cd /work/repo && bash smoke.sh" || echo "[host] smoke 有失败项"
echo "[host] env_check 已在容器启动时写入 logs/env.log"
$PD cp asr-srv:/work/repo/logs "$WORK/" >/dev/null 2>&1 || true
$PD rm -f asr-srv >/dev/null 2>&1
echo "[host] done $(date +%T); 产物 $WORK/repo/logs"
