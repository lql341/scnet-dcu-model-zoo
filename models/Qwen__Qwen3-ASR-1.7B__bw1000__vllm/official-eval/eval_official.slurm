#!/bin/bash -l
# 按官方 300 条条件执行 ASR 评测（BW1000 + vLLM）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J asr-official
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/asr-official}"
ASSETS=/public/home/scnethr2601a10/work/official-t34
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 35 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull.log 2>&1 \
  || { tail -3 /tmp/pull.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null

$PD run -d --name asr-official --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":$MODEL_PATH:ro \
  -v "$WORK":/work -v "$ASSETS":/assets:ro "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && bash env_check.sh; nohup bash run.sh > /dev/null 2>&1 & exec tail -f /dev/null" > /dev/null 2>&1

OK=0
for t in $(seq 1 120); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "[host] server up $((t*10))s"; break; }
  $PD exec asr-official bash -lc 'pgrep -f api_server >/dev/null' 2>/dev/null || { echo "[host] SERVER_PROCESS_DIED"; exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "[host] SERVER_NOT_UP"; exit 1; }

echo "[host] 官方评测开始 $(date +%T)"
$PD exec asr-official bash -lc "
cd /work/repo
export MODEL_ID='$MODEL_ID' IMAGE='$IMAGE' PORT=$PORT SERVED_NAME=$SERVED_NAME WORKERS=4
export ASSET_ROOT=/assets
mkdir -p official
python3 evaluate_official.py 2>&1 | tail -40
" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
RC=$?

echo "[host] 回收产物 $(date +%T)"
$PD cp asr-official:/work/repo/outputs "$WORK/" >/dev/null 2>&1
$PD cp asr-official:/work/repo/反馈表.csv "$WORK/" >/dev/null 2>&1
$PD cp asr-official:/work/repo/环境说明.txt "$WORK/" >/dev/null 2>&1
$PD cp asr-official:/work/repo/eval_result.json "$WORK/" >/dev/null 2>&1
$PD rm -f asr-official >/dev/null 2>&1
echo "[host] outputs 数量: $(ls "$WORK/outputs" 2>/dev/null | wc -l)"
echo "[host] DONE $(date +%T)"
