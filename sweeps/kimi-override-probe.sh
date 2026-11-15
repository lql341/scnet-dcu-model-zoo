#!/bin/bash
# 容器内只读：显式 --attention-backend 能否绕过 _get_backend_priorities 的候选集？
# 若能，FLASHMLA / ROCM_AITER_MLA 就能直接指定，不必先修 aiter。
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/override.md

{
echo '### selector.py 全文（0.20，决定 override 语义的地方）'
wc -l "$SITE"/vllm/v1/attention/selector.py 2>/dev/null
sed -n '1,40p' "$SITE"/vllm/v1/attention/selector.py 2>/dev/null
echo '...'
grep -nE 'backend_name|override|AttentionBackendEnum\[|get_valid_backends|invalid_reasons|raise|def ' \
  "$SITE"/vllm/v1/attention/selector.py 2>/dev/null | head -50
echo
echo '### get_valid_backends 完整签名'
python3 - <<'PY' 2>&1 | tail -20
import inspect
from vllm.platforms.rocm import RocmPlatform
print(inspect.signature(RocmPlatform.get_valid_backends))
print('-'*40)
print(inspect.getsource(RocmPlatform.get_valid_backends))
PY
echo
echo '### 直接测：给 use_mla=True 时各 backend 的 validate_configuration 拒绝原因'
python3 - <<'PY' 2>&1 | tail -40
from vllm.platforms import current_platform as cp
from vllm.v1.attention.backends.registry import AttentionBackendEnum
cap = cp.get_device_capability()
print('device_capability =', cap)
cfg = dict(head_size=128, dtype='bfloat16', kv_cache_dtype='auto', block_size=16,
           use_mla=True, has_sink=False, use_sparse=False, use_mm_prefix=False,
           use_per_head_quant_scales=False, use_non_causal=False, use_batch_invariant=False)
for e in AttentionBackendEnum:
    if 'MLA' not in e.name:
        continue
    try:
        cls = e.get_class()
    except Exception as ex:
        print(f'{e.name:24s} get_class FAILED {type(ex).__name__}: {str(ex)[:80]}')
        continue
    try:
        from vllm.v1.attention.selector import AttentionSelectorConfig
        c = AttentionSelectorConfig(attn_type='decoder', **cfg)
        why = cls.validate_configuration(device_capability=cap, **c._asdict())
        print(f'{e.name:24s} {"OK" if not why else why}')
    except Exception as ex:
        print(f'{e.name:24s} validate raised {type(ex).__name__}: {str(ex)[:100]}')
PY
echo
echo '### 各 MLA backend 的 CG support（是否支持 cudagraph，关系到 16K/c64 崩溃）'
python3 - <<'PY' 2>&1 | tail -20
from vllm.v1.attention.backends.registry import AttentionBackendEnum
for e in AttentionBackendEnum:
    if 'MLA' not in e.name: continue
    try:
        cls = e.get_class()
        v = cls.get_cudagraph_support()
        print(f'{e.name:24s} {v}')
    except Exception as ex:
        print(f'{e.name:24s} ? {type(ex).__name__}: {str(ex)[:70]}')
PY
} > "$OUT" 2>&1
cat "$OUT"
