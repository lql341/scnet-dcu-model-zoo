#!/bin/bash
# 显式 --attention-backend ROCM_AITER_MLA 能否真的加载起来？
# rocm.py:478 已证明它绕过 _get_backend_priorities / on_mi3xx()；
# 剩下的唯一问题是 backend 类自身是否还有 aiter 可用性检查。
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/aiter_mla.md
{
echo "### PATH 含 /opt/dtk/bin? $(case ":$PATH:" in *:/opt/dtk/bin:*) echo YES;; *) echo NO;; esac)"
export PATH=/opt/dtk/bin:$PATH
echo "### 加了之后的 ROCM_PATH/rocminfo:"
command -v rocminfo || echo "still missing"
rocminfo 2>/dev/null | grep -m2 -E 'gfx|Name:' || true
echo
echo '## 1. aiter 能否 import'
python3 -c 'import aiter; print("aiter OK", getattr(aiter,"__version__","?"))' 2>&1 | tail -5
echo
echo '## 2. is_aiter_found_and_supported / on_mi3xx'
python3 - <<'PY' 2>&1 | tail -10
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops, IS_AITER_FOUND
from vllm.platforms.rocm import on_mi3xx, _ON_MI3XX, _GCN_ARCH
print('IS_AITER_FOUND =', IS_AITER_FOUND)
print('_GCN_ARCH      =', _GCN_ARCH)
print('_ON_MI3XX      =', _ON_MI3XX)
print('on_mi3xx()     =', on_mi3xx())
print('is_aiter_found_and_supported() =', is_aiter_found_and_supported())
print('is_mla_enabled() =', rocm_aiter_ops.is_mla_enabled())
PY
echo
echo '## 3. AttentionBackendEnum.ROCM_AITER_MLA 的类路径与 get_class()'
python3 - <<'PY' 2>&1 | tail -25
from vllm.v1.attention.backends.registry import AttentionBackendEnum
e = AttentionBackendEnum.ROCM_AITER_MLA
print('enum     =', e)
print('name     =', e.name)
try:
    print('get_path() =', e.get_path())
except Exception as ex:
    print('get_path FAILED:', type(ex).__name__, ex)
try:
    c = e.get_class()
    print('get_class() =', c)
except Exception as ex:
    print('get_class FAILED:', type(ex).__name__, ex)
PY
echo
echo '## 4. backend 类源码里有没有 aiter 可用性检查'
f=$(grep -rl 'class .*AiterMLA\|ROCM_AITER_MLA' "$SITE"/vllm/v1/attention/backends/*.py 2>/dev/null | head -3)
echo "候选文件: $f"
for x in $f; do
  echo "--- $x ---"
  grep -n -E 'is_aiter|on_mi3xx|if_aiter_supported|import aiter|from aiter|raise NotImplemented|def validate_configuration' "$x" | head -20
done
echo
echo '## 5. 真正实例化：validate_configuration + __init__'
python3 - <<'PY' 2>&1 | tail -30
import torch
from vllm.platforms import current_platform as cp
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig
cfg = AttentionSelectorConfig(head_size=128, dtype=torch.bfloat16,
                              kv_cache_dtype='auto', block_size=16, use_mla=True)
cap = cp.get_device_capability()
print('capability =', cap)
e = AttentionBackendEnum.ROCM_AITER_MLA
try:
    cls = e.get_class()
    print('class =', cls)
    print('supports_mla =', getattr(cls,'supported_kv_cache_dtypes',None))
    r = cls.validate_configuration(device_capability=cap, **cfg._asdict())
    print('validate_configuration ->', r if r else 'OK (empty)')
except Exception as ex:
    import traceback; traceback.print_exc()
PY
echo
echo '## 6. 两个 aiter MLA 都试一遍'
python3 - <<'PY' 2>&1 | tail -25
import torch
from vllm.platforms import current_platform as cp
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E
from vllm.v1.attention.selector import AttentionSelectorConfig
cfg = AttentionSelectorConfig(head_size=128, dtype=torch.bfloat16,
                              kv_cache_dtype='auto', block_size=16, use_mla=True)
cap = cp.get_device_capability()
for name in ('ROCM_AITER_MLA','ROCM_AITER_TRITON_MLA','TRITON_MLA'):
    e = getattr(E, name)
    try:
        cls = e.get_class()
        r = cls.validate_configuration(device_capability=cap, **cfg._asdict())
        print(f'{name:24s} get_class OK  validate ->', r if r else 'OK (empty)')
    except Exception as ex:
        print(f'{name:24s} FAILED {type(ex).__name__}: {str(ex)[:160]}')
PY
} > "$OUT" 2>&1
cat "$OUT"
