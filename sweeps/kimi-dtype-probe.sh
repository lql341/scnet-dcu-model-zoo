#!/bin/bash
# 容器内只读：用**正确的 torch.dtype** 重测各 MLA backend 的可用性
# 上一版把 dtype 传成了字符串 'bfloat16'，与 supported_dtypes 里的 torch.bfloat16 不等，
# 导致全部误报 "dtype not supported"。
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/dtype.md
{
echo '### 各 backend 的 supported_dtypes'
python3 - <<'PY' 2>&1 | tail -20
from vllm.v1.attention.backends.registry import AttentionBackendEnum
for e in AttentionBackendEnum:
    if 'MLA' not in e.name: continue
    try: cls = e.get_class()
    except Exception: continue
    if hasattr(cls, 'supported_dtypes'):
        print(f'{e.name:24s} {cls.supported_dtypes}')
PY
echo
echo '### 用 torch.bfloat16 重测（Kimi 的真实 dtype）'
python3 - <<'PY' 2>&1 | tail -60
import torch
from vllm.platforms import current_platform as cp
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import AttentionSelectorConfig

cap = cp.get_device_capability()
print('device_capability =', cap)

# Kimi linear_attn_config: head_dim=128（KDA 层），MLA 层 qk_nope 128 + qk_rope 64
for head_size in (128, 192, 72):
    print(f'--- head_size={head_size} dtype=torch.bfloat16 ---')
    cfg = AttentionSelectorConfig(
        head_size=head_size, dtype=torch.bfloat16, kv_cache_dtype='auto',
        block_size=16, use_mla=True, has_sink=False, use_sparse=False,
        use_mm_prefix=False, use_per_head_quant_scales=False,
        use_non_causal=False, use_batch_invariant=False)
    for e in AttentionBackendEnum:
        if 'MLA' not in e.name: continue
        try:
            cls = e.get_class()
        except Exception as ex:
            print(f'  {e.name:24s} get_class FAILED: {type(ex).__name__}: {str(ex)[:70]}')
            continue
        try:
            why = cls.validate_configuration(device_capability=cap, **cfg._asdict())
            print(f'  {e.name:24s} {"✅ OK" if not why else why}')
        except Exception as ex:
            print(f'  {e.name:24s} raised {type(ex).__name__}: {str(ex)[:80]}')
PY
} > "$OUT" 2>&1
cat "$OUT"
