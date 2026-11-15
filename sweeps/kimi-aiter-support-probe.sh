#!/bin/bash
# 为什么 import aiter 成功、is_aiter_found_and_supported() 仍为 False
# 输出 _aiter_ops.py 的 25-105 行（两个判定函数 + 装饰器），并逐个试触发条件
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/support.md
export PATH=/tmp/fakebin:$PATH
mkdir -p /tmp/fakebin
printf '#!/bin/bash\necho "  gfx: gfx936"\n' > /tmp/fakebin/rocminfo
chmod +x /tmp/fakebin/rocminfo
{
echo '### _aiter_ops.py 1..105'
sed -n '1,105p' "$SITE"/vllm/_aiter_ops.py
echo
echo '### rocm.py is_aiter_available（350 行上方那个函数）'
sed -n '330,362p' "$SITE"/vllm/platforms/rocm.py
echo
echo '### 逐个问判定条件'
python3 - <<'PY' 2>&1 | tail -40
import os
for k in sorted(os.environ):
    if 'AITER' in k or 'ROCM' in k or 'HIP' in k or 'VLLM' in k:
        print('ENV', k, '=', os.environ[k])
print('---')
import aiter
print('aiter file:', aiter.__file__)
for attr in ('__version__','get_gfx','is_aiter_supported','AITER_AVAILABLE','has_cpp'):
    print(attr, '=', getattr(aiter, attr, 'MISSING'))
PY
echo
echo '### 关键：vllm 里 aiter 的支持判定到底查什么'
grep -n -A25 'def is_aiter_found_and_supported' "$SITE"/vllm/_aiter_ops.py
echo '--- is_aiter_found ---'
grep -n -A15 '^def is_aiter_found' "$SITE"/vllm/_aiter_ops.py
echo '--- rocm_aiter_ops.is_mla_enabled 定义处 ---'
grep -rn -B3 -A12 'def is_mla_enabled' "$SITE"/vllm/_aiter_ops.py
echo
echo '### 直接调用（假 rocminfo 在 PATH 上）'
python3 - <<'PY' 2>&1 | tail -25
import traceback
try:
    import vllm._aiter_ops as m
    print('IS_AITER_FOUND =', m.IS_AITER_FOUND)
    print('is_aiter_found_and_supported() =', m.is_aiter_found_and_supported())
    ops = m.rocm_aiter_ops
    for n in ('is_enabled','is_mla_enabled','is_mha_enabled','is_linear_fp8_enabled','is_fused_moe_enabled'):
        f = getattr(ops, n, None)
        try: print(n, '=', f() if f else 'MISSING')
        except Exception as e: print(n, 'raised', type(e).__name__, e)
except Exception:
    traceback.print_exc()
PY
} > "$OUT" 2>&1
cat "$OUT"
