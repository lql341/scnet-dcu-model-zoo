#!/bin/bash -l
# HY-OmniWeaving 探针：装隔离依赖 → 导入校验 → 跑 1 条官方条件（81 帧 / 480p / 50 步）
# 目的：先确认「能不能起、显存够不够」，再决定 60 条怎么排（单卡串行 / 多卡 FSDP）
#   sbatch sweeps/omniweaving-probe.slurm
#   FRAMES=17 STEPS=8 sbatch sweeps/omniweaving-probe.slurm     # 更轻的管线验证
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J omni-probe
#SBATCH -o /public/home/scnethr2601a10/work/omni-probe-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/omni-probe-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/HY-OmniWeaving
SRCHOST=/public/home/scnethr2601a10/OmniWeaving-src
WORK=/public/home/scnethr2601a10/work/omni-probe
FRAMES=${FRAMES:-81}; STEPS=${STEPS:-50}
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G frames=$FRAMES steps=$STEPS"
# SGLang 镜像解包约 32 GB，留 45 G 足够（60 G 门槛在拥挤时段几乎排不上）
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
[ -s "$TAR" ] && $PD load -i "$TAR" >/tmp/load.log 2>&1
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/Omni:ro -v "$SRCHOST":/omni-src:ro -v "$WORK":/work \
  -v /public/home/scnethr2601a10/wheels/omni-fix:/wheels:ro \
  -v /public/home/scnethr2601a10/work/omni-patches:/patches:ro \
  -e FRAMES="$FRAMES" -e STEPS="$STEPS" \
  "$IMG" -c '
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo "--- 隔离安装依赖（--no-deps，cp310）---"
n=0; f=0
for w in /wheels/*.whl; do
  if pip install -q --no-deps --target /opt/omni-libs "$w" 2>/dev/null; then n=$((n+1)); else f=$((f+1)); echo "  [warn] fail: $(basename "$w")"; fi
done
echo "已装 $n，失败 $f"
export PYTHONPATH=/opt/omni-libs:/omni-src:${PYTHONPATH:-}
echo "--- 版本与导入 ---"
python3 - <<PYCHK
import importlib, sys
for m in ("torch","torchvision","transformers","diffusers","huggingface_hub","peft","einops","loguru","imageio","omegaconf","qwen_vl_utils","decord","cv2"):
    try:
        mod = importlib.import_module(m); print(f"  {m:<16} {getattr(mod,\"__version__\",\"?\")}")
    except Exception as e:
        print(f"  {m:<16} FAIL {type(e).__name__}: {e}")
PYCHK
python3 -c "import hyvideo; print(\"hyvideo import OK\")" 2>&1 | tail -5
echo "--- 打 DCU 兼容补丁（flash_attn unpad 4 项 vs 上游按 5 项解包）---"
cp -r /omni-src /opt/omni-src
for p in /patches/*.patch; do
  [ -e "$p" ] || continue
  echo "  apply $(basename "$p")"; patch -p1 -d /opt/omni-src < "$p" || exit 6
done
export PYTHONPATH=/opt/omni-libs:/opt/omni-src:${PYTHONPATH:-}
echo "--- 跑 1 条（frames=$FRAMES steps=$STEPS）---"
mkdir -p /work/out
cd /opt/omni-src && timeout 5400 python3 generate.py \
  --prompt "a photo of a bench, cinematic lighting" \
  --model_path /models/Omni \
  --task t2v --resolution 480p --aspect_ratio 16:9 \
  --video_length "$FRAMES" --num_inference_steps "$STEPS" --total_steps "$STEPS" \
  --seed 42 --dtype bf16 --offloading true \
  --output_path /work/out/probe.mp4 2>&1 | tail -30
echo "--- 产物 ---"; ls -la /work/out 2>/dev/null
' 2>&1 | grep -viE "subuid|rootless|level=warning|^Copying|^Getting"
echo "[host] DONE $(date +%T)"
