#!/bin/bash -l
# AuK × BW1000：先验证能否在镜像 torch 上跑通 1 条，再决定是否全量 120 条
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J auk-probe
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/auk}"
ASSETS=/public/home/scnethr2601a10/work/official-t34
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 35 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull.log 2>&1 || { tail -3 /tmp/pull.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null

$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$HOST_ENCODER_PATH":$ENCODER_PATH:ro \
  -v "$WORK":/work -v "$ASSETS":/assets:ro -v "$WHEELS_HOST":/wheels:ro -v "$AUK_SRC_HOST":/auk-src:ro \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo '--- 镜像内版本 ---'
python3 -c 'import torch,transformers;print(\"torch\",torch.__version__,\"transformers\",transformers.__version__)'
echo '--- 装 AuK（--no-deps，忽略 torch/transformers 的 pin）---'
pip install -q --no-deps --target /opt/auk-libs /wheels/*.whl 2>&1 | tail -2
# auk 是 pyproject 项目，直接 pip --target 装不进；把源码 src 目录加进 PYTHONPATH 更稳
export PYTHONPATH=/opt/auk-libs:/auk-src/src:\$PYTHONPATH
ls /auk-src/src/auk | head -8
echo '--- 导入测试 ---'
python3 -c 'import auk; print(\"auk import OK\", getattr(auk,\"__version__\",\"?\"))' 2>&1 | tail -5
python3 -c 'from auk.infer.infer_cli import main; print(\"ckpt cli OK\")' 2>&1 | tail -5
echo '--- 单条推理试跑 ---'
mkdir -p /work/out
python3 /work/repo/probe_one.py 2>&1 | tail -25
" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] DONE $(date +%T)"
