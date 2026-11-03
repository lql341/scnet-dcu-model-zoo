#!/usr/bin/env bash
set -uo pipefail
source ./env.sh
mkdir -p logs
{
  echo "image: $IMAGE"
  hy-smi 2>/dev/null | head -12
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0); print('Device:', p.name, 'vram_GB:', round(p.total_memory/2**30,1), 'arch:', p.gcnArchName)
"
  echo "-- 隔离目录内的 transformers --"
  PYTHONPATH=/opt/fa-libs python3 -c "import transformers,sys;print('transformers(isolated):',transformers.__version__)" 2>/dev/null || echo "transformers(isolated): 未安装（需先跑 container/eval_official.slurm）"
  PYTHONPATH=/opt/fa-libs python3 -c "import qwen_asr;print('qwen_asr: OK')" 2>/dev/null || echo "qwen_asr: 未安装"
  python3 -c "import vllm;print('vllm(镜像内):',vllm.__version__)" 2>/dev/null
} 2>&1 | tee logs/env.log
