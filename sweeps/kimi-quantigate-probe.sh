#!/bin/bash
# P0-B：0.20 / 0.21 两个镜像的量化白名单与 fp8 判定（0 卡时，不加载模型）
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
tag="${1:-unknown}"
OUT=/gate/quant-$tag.md
{
echo "### image=$tag  site=$SITE"
echo
echo '## 1. 当前生效的平台对象'
python3 - <<'PY' 2>&1 | tail -30
import vllm
from vllm.platforms import current_platform as cp
print('vllm', vllm.__version__)
print('platform class:', type(cp).__name__)
print('platform module:', type(cp).__module__)
for name in ('supported_quantization','get_device_capability','is_fp8_supported',
             'get_supported_attn_backends','supported_attention_backends','fp8_dtype'):
    v = getattr(cp, name, None)
    if v is None:
        print(f'{name}: MISSING'); continue
    try:
        print(f'{name} =', v() if callable(v) else v)
    except Exception as e:
        print(f'{name}: raised {type(e).__name__}: {e}')
PY
echo
echo '## 2. 量化方法注册表（实际能用的名字）'
python3 - <<'PY' 2>&1 | tail -25
try:
    from vllm.model_executor.layers.quantization import get_quantization_config
    names = ['fp8','fp8_e4m3','fp8_e5m2','fp8_w8a8','fp8_per_tensor','awq','gptq',
             'compressed-tensors','int8','w8a8_int8','fp8_marlin','mxfp4','w4a8']
    for n in names:
        try:
            cls = get_quantization_config(n)
            print(f'{n:24s} -> {cls.__name__ if cls else "None"}')
        except Exception as e:
            print(f'{n:24s} !! {type(e).__name__}: {str(e)[:90]}')
except Exception as e:
    print('registry import failed:', type(e).__name__, e)
PY
echo
echo '## 3. vllm_hcu 的量化白名单（0.21 才有）'
grep -rn --include='*.py' -E 'supported_quantization|fp8|FP8' "$SITE"/vllm_hcu/ 2>/dev/null | head -40
echo
[ -d "$SITE/vllm_hcu" ] && echo "[vllm_hcu 存在]" || echo "[无 vllm_hcu 目录——这是 0.20]"
echo
echo '## 4. 0.20 的 RocmPlatform.supported_quantization 原文'
grep -n -A30 'supported_quantization' "$SITE"/vllm/platforms/rocm.py 2>/dev/null | head -40
echo
echo '## 5. hipBLASLt 的 fp8 符号'
ls -l /opt/dtk/lib/libhipblaslt* 2>/dev/null
strings /opt/dtk/lib/libhipblaslt.so* 2>/dev/null | grep -iE '^fp8|e4m3|e5m2' | sort -u | head -15
echo
echo '## 6. fp8 kernel 实际能否跑（最小 GEMM）'
python3 - <<'PY' 2>&1 | tail -25
import torch
print('torch', torch.__version__, 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
for dt in (torch.float8_e4m3fn, torch.float8_e5m2):
    try:
        a = torch.zeros(16,16, device='cuda', dtype=dt)
        b = torch.zeros(16,16, device='cuda', dtype=dt)
        c = torch.mm(a,b)
        print(type(dt).__name__, dt, 'mm OK ->', c.dtype)
    except Exception as e:
        print(dt, 'mm FAIL', type(e).__name__, str(e)[:140])
PY
} > "$OUT" 2>&1
cat "$OUT"
