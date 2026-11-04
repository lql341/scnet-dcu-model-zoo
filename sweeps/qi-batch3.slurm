#!/bin/bash -l
# Qwen-Image 合批优化：同一 prompt 的 3 个 seed 合成一次调用（对比逐张生成的 19.8 s/张）
#   sbatch sweeps/qi-batch3.slurm                      # 全量 60 prompt / 180 张
#   PROBE_PROMPTS=8 sbatch sweeps/qi-batch3.slurm      # 先跑 8 个 prompt 探路
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J qi-batch3
#SBATCH -o /public/home/scnethr2601a10/work/qi-batch3-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/qi-batch3-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/Qwen-Image-2.1-Turbo
WORK=/public/home/scnethr2601a10/work/qi-batch3
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 50 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
[ -s "$TAR" ] && $PD load -i "$TAR" >/tmp/load.log 2>&1
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }
mkdir -p "$WORK"/{repo,bench-batch}
COND=/public/home/scnethr2601a10/work/official-conds/qwen_image.jsonl
[ -s "$COND" ] || { echo "[host] 缺 $COND"; exit 3; }
cp -f "$COND" "$WORK/repo/qwen_image.jsonl"
cp -f /public/home/scnethr2601a10/work/qi-batch3-src/batch3.py "$WORK/repo/" || { echo "[host] 缺 batch3.py"; exit 3; }
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/QI:ro -v "$WORK":/work -v /public/home/scnethr2601a10/wheels/qi2:/wheels:ro \
  -e PROBE_PROMPTS="${PROBE_PROMPTS:-0}" \
  "$IMG" -c '
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
pip install -q --no-deps --target /opt/qi-libs /wheels/diffusers-0.41.0-py3-none-any.whl /wheels/huggingface_hub-1.33.0-py3-none-any.whl 2>&1 | tail -1
export PYTHONPATH=/opt/qi-libs:${PYTHONPATH:-}
cd /work/repo
MODEL_PATH=/models/QI COND_FILE=qwen_image.jsonl OUTDIR=/work/bench-batch/outputs \
  PROBE_PROMPTS="$PROBE_PROMPTS" python3 batch3.py 2>&1 | tail -25
' 2>&1 | grep -viE "subuid|rootless|level=warning|^Copying|^Getting"
echo "[host] DONE $(date +%T)"; ls "$WORK/bench-batch/outputs" 2>/dev/null | wc -l
