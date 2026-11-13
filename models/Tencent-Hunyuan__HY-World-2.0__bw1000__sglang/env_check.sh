#!/usr/bin/env bash
# 采集环境 → logs/env.log（容器内执行；harness 的 env_check.sh 会打印更详细的信息）
set -uo pipefail
source ./env.sh 2>/dev/null || true
mkdir -p logs
{
  echo "image: $IMAGE"
  echo "model: $MODEL_ID  (WorldMirror 2.0 子集)"
  hy-smi 2>/dev/null | head -8
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print('Device:', p.name, 'vram_GB:', round(p.total_memory/2**30,1), 'arch:', p.gcnArchName)
"
  echo "-- 依赖（镜像自带 /opt/hyworld2-bw/deps）--"
  ls /opt/hyworld2-bw/deps 2>/dev/null | head -8
  echo "-- gsplat（镜像内已编译）--"
  find / -name "csrc.so" -path "*gsplat*" 2>/dev/null | head -2
} 2>&1 | tee logs/env.log
