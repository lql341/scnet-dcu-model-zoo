#!/bin/bash -l
# Qwen3-ForcedAligner-0.6B × BW1000 × Transformers：按官方 30 条英文条件做词级对齐
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J fa-official
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/fa-official}"
ASSETS=/public/home/scnethr2601a10/work/official-t34
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 35 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull-$JID.log 2>&1 \
  || { tail -3 /tmp/pull-$JID.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null

echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":$MODEL_PATH:ro \
  -v "$WORK":/work -v "$ASSETS":/assets:ro -v "$WHEELS_HOST":/wheels:ro \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo '--- qwen-asr 依赖 transformers==4.57.6，与镜像的 5.8.1 不兼容（check_model_inputs 签名变了）---'
echo '--- 方案：把 4.57.6 装到独立目录 /opt/fa-libs，用 PYTHONPATH 前置，不动镜像本体 ---'
mkdir -p /opt/fa-libs
# transformers 4.57.6 还要求 huggingface-hub<1.0（镜像内是 1.11.0）
pip install -q --no-deps --target /opt/fa-libs /wheels/transformers-4.57.6-py3-none-any.whl \
  /wheels/huggingface_hub-*.whl /wheels/tokenizers-*.whl 2>&1 | tail -2
# nagisa 依赖 dyNET（重 C++），而它只用于日语分词；本条目评测集为 30 条英文，
# 故用最小 stub 替换，避免为不触达的路径引入重型依赖。
pip install -q --no-deps --target /opt/fa-libs /wheels/qwen_asr-*.whl /wheels/soynlp-*.whl /wheels/qwen_omni_utils-*.whl 2>&1 | tail -2
cp /work/repo/wheels/stub/nagisa.py /opt/fa-libs/nagisa.py
echo 'nagisa -> stub (English-only; Japanese path raises)'
export PYTHONPATH=/opt/fa-libs:\$PYTHONPATH
python3 -c 'import transformers;print(\"transformers-isolated\", transformers.__version__)' 2>&1 | tail -2
python3 -c 'import qwen_asr; print(\"qwen_asr OK\")' 2>&1 | tail -3
python3 -c 'import transformers,vllm;print(\"transformers\",transformers.__version__,\"| vllm\",vllm.__version__)' 2>&1 | tail -1
cd /work/repo
export MODEL_ID='$MODEL_ID' MODEL_PATH='$MODEL_PATH' IMAGE='$IMAGE'
export COND_FILE=conditions/forced_aligner.jsonl ASSET_ROOT=/assets OUTDIR=outputs
python3 evaluate_official.py 2>&1 | tail -35
" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"

echo "[host] 产物 $(date +%T)"
echo "[host] outputs: $(ls "$WORK/repo/outputs" 2>/dev/null | wc -l) 个文件"
echo "[host] DONE $(date +%T)"
