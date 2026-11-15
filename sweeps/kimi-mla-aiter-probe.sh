#!/bin/bash
# 容器内只读：为什么 0.20 上 Kimi 的 MLA 候选集只有 TRITON_MLA（而不是 aiter 三件套）
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/mla.md

{
echo '### rocm.py 350-400'
sed -n '350,400p' "$SITE"/vllm/platforms/rocm.py 2>/dev/null
echo
echo '### _aiter_ops.py：MLA 相关函数全文'
grep -nE 'def |^[A-Z_]+ *=' "$SITE"/vllm/_aiter_ops.py 2>/dev/null | head -40
echo
echo '--- is_mla_enabled ---'
sed -n "$(grep -n 'def is_mla_enabled' "$SITE"/vllm/_aiter_ops.py | cut -d: -f1),+25p" "$SITE"/vllm/_aiter_ops.py 2>/dev/null
echo
echo '--- is_aiter_found_and_supported ---'
sed -n "$(grep -n 'def is_aiter_found_and_supported' "$SITE"/vllm/_aiter_ops.py | cut -d: -f1),+30p" "$SITE"/vllm/_aiter_ops.py 2>/dev/null
echo
echo '### 实际调用结果'
python3 - <<'PY' 2>&1 | tail -40
try:
    from vllm._aiter_ops import rocm_aiter_ops, is_aiter_found_and_supported
    for n in ('is_enabled','is_mla_enabled','is_mha_enabled','is_linear_fp8_enabled','is_rmsnorm_enabled','is_fused_moe_enabled'):
        fn = getattr(rocm_aiter_ops, n, None)
        try:
            print(n, '=', fn() if fn else 'MISSING')
        except Exception as e:
            print(n, 'raised', type(e).__name__, e)
    print('is_aiter_found_and_supported =', is_aiter_found_and_supported())
except Exception as e:
    import traceback; traceback.print_exc()
PY
echo
echo '### 直接问：MLA 的候选集（模拟 Kimi 的 selector 输入）'
python3 - <<'PY' 2>&1 | tail -50
import inspect
try:
    from vllm.platforms import current_platform as cp
    from vllm.v1.attention.selector import AttentionSelectorConfig
    sig = inspect.signature(AttentionSelectorConfig)
    print('AttentionSelectorConfig fields:', list(sig.parameters))
    cfg = AttentionSelectorConfig(**{k: (v.default if v.default is not inspect._empty else None) for k, v in sig.parameters.items()})
    print('default cfg:', cfg)
    for use_mla in (True,):
        for use_sparse in (False, True):
            c = cfg._replace(use_mla=use_mla, use_sparse=use_sparse)
            try:
                valid, invalid = cp.get_valid_backends(c, device_capability=cp.get_device_capability())
                print(f'use_mla={use_mla} sparse={use_sparse} -> valid={[v[0].name for v in valid]}')
                for b, why in invalid.items():
                    print(f'    invalid {b.name}: {why}')
            except Exception as e:
                print('get_valid_backends raised', type(e).__name__, e)
except Exception as e:
    import traceback; traceback.print_exc()
PY
echo
echo '### _get_backend_priorities 源码'
sed -n "$(grep -n 'def _get_backend_priorities' "$SITE"/vllm/platforms/rocm.py | cut -d: -f1),+45p" "$SITE"/vllm/platforms/rocm.py 2>/dev/null
echo
echo '### attention selector 里对 use_mla / override backend 的处理'
grep -nE 'use_mla|backend_name|attention_backend|get_valid_backends|selected_backend' "$SITE"/vllm/v1/attention/selector.py 2>/dev/null | head -40
} > "$OUT" 2>&1
cat "$OUT"
