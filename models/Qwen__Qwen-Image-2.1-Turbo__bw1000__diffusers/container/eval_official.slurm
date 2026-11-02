#!/bin/bash -l
# Qwen-Image-2.1-Turbo × BW1000 × Diffusers：官方 180 条（2048×2048）
# 先跑 3 条探针验证管线，PROBE_ONLY=1 时只跑前 3 条
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J qi-official
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/qi}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then $PD load -i "$HOST_IMAGE_TAR" >/tmp/load.log 2>&1; fi
$PD image exists "$IMAGE" 2>/dev/null || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/load.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/load.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":$MODEL_PATH:ro \
  -v "$WORK":/work -v "$WHEELS_HOST":/wheels:ro \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo '--- 镜像内版本 ---'
python3 -c 'import torch,transformers;print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__)'
echo '--- 装 diffusers（--no-deps，不动镜像 torch/transformers）---'
pip install -q --no-deps --target /opt/qi-libs /wheels/diffusers-*.whl 2>&1 | tail -2
export PYTHONPATH=/opt/qi-libs:\$PYTHONPATH
python3 -c 'import diffusers;print(\"diffusers\",diffusers.__version__);from diffusers import QwenImage21Pipeline;print(\"QwenImage21Pipeline OK\")' 2>&1 | tail -4
cd /work/repo
export MODEL_PATH='$MODEL_PATH' WIDTH=$WIDTH HEIGHT=$HEIGHT
export COND_FILE=conditions/qwen_image.jsonl OUTDIR=outputs PROBE_ONLY=\${PROBE_ONLY:-0}
python3 evaluate_official.py 2>&1 | tail -30
" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] outputs: $(ls "$WORK/repo/outputs" 2>/dev/null | wc -l) 个"
echo "[host] DONE $(date +%T)"
