#!/bin/bash
# 第二道门：is_aiter_found_and_supported() -> on_mi3xx()
# 只看源码，不 import torch/aiter（上一发 import 阶段疑似卡死）
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/gate2.md
{
echo '### on_mi3xx / on_gfx1x 定义'
grep -n -A14 '^def on_mi3xx' "$SITE"/vllm/platforms/rocm.py
echo '--- 引用点 ---'
grep -n 'on_mi3xx' "$SITE"/vllm/platforms/rocm.py
echo
echo '### device capability 怎么算（is_mi3xx 之类）'
grep -n -B2 -A10 'mi3xx\|MI300\|gfx94' "$SITE"/vllm/platforms/rocm.py | head -40
echo
echo '### _MLA_ENABLED / _AITER_ENABLED 赋值'
grep -n -B6 -A6 '_MLA_ENABLED\s*=' "$SITE"/vllm/_aiter_ops.py | head -40
grep -n -B4 -A4 '_AITER_ENABLED\s*=' "$SITE"/vllm/_aiter_ops.py | head -30
echo
echo '### 本机 arch（纯 torch，不碰 aiter）'
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_properties(0))" 2>&1 | tail -5
} > "$OUT" 2>&1
cat "$OUT"
