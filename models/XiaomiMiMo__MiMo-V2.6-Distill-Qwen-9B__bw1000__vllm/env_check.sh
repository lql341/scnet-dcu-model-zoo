#!/usr/bin/env bash
# 采集验证环境写入 env.log；报告「2. 验证环境」的版本表直接取自这里。
# 在容器内执行（容器外宿主机无 torch / sglang）。
set -uo pipefail
source ./env.sh
mkdir -p logs

{
  echo "image: $IMAGE"
  echo "card:  $CARD   framework: $FRAMEWORK   precision: $PRECISION"
  echo "dtk:   $(cat /opt/dtk/.info/version 2>/dev/null || echo "DTK_VERSION=$DTK_VERSION")"
  echo
  echo "== device =="
  ls -l /dev/kfd /dev/mkfd 2>/dev/null
  /opt/hyhal/bin/hy-smi 2>/dev/null || hy-smi 2>/dev/null || echo "hy-smi not found"
  echo
  echo "== python / torch =="
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('Device:', p.name)
    print('vram_GB:', round(p.total_memory / 2**30, 1))
    print('arch:', getattr(p, 'gcnArchName', '?'))
"
  echo
  echo "== components =="
  for m in vllm triton transformers tokenizers; do
    python3 -c "import $m; print('$m:', getattr($m, '__version__', '?'))" 2>/dev/null
  done
  echo
  echo "== pip list (key) =="
  python3 -m pip list 2>/dev/null | grep -iE "^(torch|triton|sglang|transformers|tokenizers|flash|fla)" | head -20
} 2>&1 | tee logs/env.log
