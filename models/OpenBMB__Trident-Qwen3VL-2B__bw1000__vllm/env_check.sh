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
  python3 -c "import vllm,transformers,triton;print('vllm:',vllm.__version__);print('transformers:',transformers.__version__);print('triton:',triton.__version__)" 2>&1|tail -3
  python3 -c "import soundfile, librosa" 2>&1 | tail -1 || true
} 2>&1 | tee logs/env.log
