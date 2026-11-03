#!/bin/bash -l
# W2 三个模型的架构支持探测：镜像内是否有对应实现 + config.json 的 architectures
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J w2-probe
#SBATCH -o /public/home/scnethr2601a10/work/w2/probe-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/w2/probe-%j.err
set -uo pipefail
IMG_SGL=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
IMG_VLLM=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
M=/public/home/scnethr2601a10/models
W=/public/home/scnethr2601a10/work/w2
mkdir -p "$W"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

echo "== config.json 里的 architectures =="
for d in Recon2Reason-Reasoning-4B Trident-Qwen3VL-2B RoboBrain2.5-8B-NV; do
  printf "  %-28s " "$d"
  python3 -c "
import json
c=json.load(open('$M/$d/config.json'))
print(c.get('architectures'), '| model_type:', c.get('model_type'), '| dtype:', c.get('torch_dtype') or c.get('dtype'))
" 2>&1 | head -1
done

for I in "$IMG_SGL" "$IMG_VLLM"; do
  echo ""
  echo "########## $(basename $(echo $I | cut -d/ -f6))"
  $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$I" >/dev/null 2>&1 || { echo "PULL_FAIL"; continue; }
  $PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro "$I" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
echo '--- 框架版本 ---'
python3 -c 'import sglang;print(\"sglang:\",sglang.__version__)' 2>/dev/null || \
python3 -c 'import vllm;print(\"vllm:\",vllm.__version__)' 2>/dev/null
echo '--- 模型实现文件 ---'
for pat in qwen3_vl qwen3vl kimi_vl qwen2_5_vl; do
  ls /usr/local/lib/python3.10/dist-packages/*/model_executor/models/*\${pat}* 2>/dev/null | head -2
  ls /usr/local/lib/python3.10/dist-packages/sglang/srt/models/*\${pat}* 2>/dev/null | head -2
done
echo '--- transformers 支持这三种 config 吗 ---'
python3 -c '
import transformers as T
for a in [\"Qwen3VLForConditionalGeneration\",\"Qwen3VLMoeForConditionalGeneration\",\"Qwen3VLForEmbedding\"]:
    print(\"  \", a, \"->\", \"YES\" if hasattr(T, a) else \"no\")
' 2>&1 | grep -E 'YES|no'
" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
done
echo "== DONE $(date +%T) =="
