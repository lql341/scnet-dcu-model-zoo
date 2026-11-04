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
  python3 -c "import transformer" 2>/dev/null || python3 -c "import transformers;print('transformers(镜像内):',transformers.__version__)"
  python3 -c "import diffusers;print('diffusers(镜像内):',diffusers.__version__)" 2>/dev/null || echo "diffusers(镜像内): 无（条目从 /opt/qi-libs 隔离安装 0.41.0）"
  python3 -c "import flash_attn;print('flash_attn:',flash_attn.__version__)" 2>/dev/null || echo "flash_attn: 无 → wan 回落 torch SDPA（预期）"
  echo "-- 官方代码 --"
  ls "$WAN_SRC_HOST" 2>/dev/null | head -5
} 2>&1 | tee logs/env.log
