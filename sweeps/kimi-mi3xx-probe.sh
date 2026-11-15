#!/bin/bash
# 关键问题：显式 --attention-backend ROCM_AITER_MLA 能否绕过 on_mi3xx()？
# 只读（不加载模型），回答三件事：
#   ① on_mi3xx() 在本机返回什么、判据是什么
#   ② BackendSelector 显式指定时是否还过 validate_configuration / @if_aiter_supported
#   ③ 显式指定后 get_valid_backends 的实际结果
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/mi3xx.md
{
echo '### 0. on_mi3xx 定义'
grep -n -A16 '^def on_mi3xx' "$SITE"/vllm/platforms/rocm.py
echo '--- on_gfx1x / on_mi300 / 相关 ---'
grep -n -A10 '^def on_gfx1x\|^def on_mi3xx\|def is_mi3' "$SITE"/vllm/platforms/rocm.py | head -50
echo
echo '### 1. 显式指定 backend 的代码路径（selector）'
grep -n -B5 -A30 'attention_backend.*is not None\|explicit' "$SITE"/vllm/v1/attention/selector.py 2>/dev/null | head -60
echo '--- BackendSelector.select 关键分支 ---'
grep -n -A8 'def select\b' "$SITE"/vllm/v1/attention/selector.py 2>/dev/null | head -40
echo
echo '### 2. 运行期实测'
python3 - <<'PY' 2>&1 | tail -40
import traceback, torch
try:
    from vllm.platforms.rocm import on_mi3xx
    print('on_mi3xx() =', on_mi3xx())
except Exception as e:
    print('on_mi3xx import failed:', e)
try:
    from vllm.platforms import current_platform as cp
    print('platform =', type(cp).__name__)
    print('capability =', cp.get_device_capability())
    from vllm.v1.attention.selector import AttentionSelectorConfig, get_attn_backend
    print('selector exports OK')
except Exception:
    traceback.print_exc()
try:
    import vllm.platforms.rocm as R
    print('rocm module funcs:', [n for n in dir(R) if n.startswith(('on_','is_'))][:20])
except Exception as e:
    print('rocm dir failed', e)
PY
echo
echo '### 3. 显式 backend 时 validate_configuration 还跑不跑'
grep -n -B4 -A20 'validate_configuration' "$SITE"/vllm/v1/attention/selector.py 2>/dev/null | head -50
} > "$OUT" 2>&1
cat "$OUT"
