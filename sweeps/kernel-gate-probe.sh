#!/bin/bash
# 在 vLLM 容器内执行：只读探测，不加载模型
# 由 kernel-gate-2026-11-15.slurm 挂载到 /gate/gate_probe.sh 并调用
set -uo pipefail
TAG="${1:?usage: gate_probe.sh <tag>}"
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/$TAG.md

{
echo "##### $TAG #####"
echo
echo '### Q0 versions'
python3 -c 'import vllm;print("vllm",vllm.__version__)' 2>&1 | tail -1
python3 -c 'import torch;print("torch",torch.__version__,torch.version.hip)' 2>&1 | tail -1
python3 -c 'import sys;print("python",sys.version.split()[0])' 2>&1 | tail -1
echo
echo '### Q1 attention backend enum + rocm platform'
python3 - <<'PY' 2>&1 | tail -50
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum
    print('AttentionBackendEnum:', [e.name for e in AttentionBackendEnum])
except Exception as e:
    print('registry import failed:', type(e).__name__, e)
try:
    from vllm.platforms import rocm as r
    cands = [a for a in dir(r.RocmPlatform) if 'backend' in a.lower() or 'attn' in a.lower()]
    print('RocmPlatform attn/backend attrs:', cands)
    for n in cands:
        fn = getattr(r.RocmPlatform, n)
        if callable(fn):
            try:
                print('  %s() =' % n, fn())
            except Exception as e:
                print('  %s() raised' % n, type(e).__name__, e)
except Exception as e:
    print('rocm platform import failed:', type(e).__name__, e)
PY
echo
echo '### Q1b backend module files'
ls "$SITE"/vllm/v1/attention/backends/*.py 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '; echo
echo
echo '### Q2 aiter'
python3 -c 'import aiter;print("aiter",aiter.__version__)' 2>&1 | tail -2
ls -d "$SITE"/aiter 2>/dev/null
python3 -c 'import aiter;print("aiter exports:",[x for x in dir(aiter) if not x.startswith("_")][:40])' 2>&1 | tail -2
echo
echo '### Q3 hipBLASLt fp8'
ls -l /opt/dtk/lib/libhipblaslt* 2>/dev/null
ls -d "$SITE"/hipblaslt* 2>/dev/null | head -3
python3 -c 'import hipblaslt;print("hipblaslt module ok")' 2>&1 | tail -2
/opt/dtk/bin/hipblaslt-bench --help 2>&1 | grep -iE 'fp8|e4m3|f8' | head -5
strings /opt/dtk/lib/libhipblaslt.so* 2>/dev/null | grep -oE 'gfx936[^ ]*' | sort -u | head -5
echo
echo '### Q4 Kimi-Linear kernel path'
ls "$SITE"/vllm/model_executor/models/ 2>/dev/null | grep -iE 'kimi|linear' | tr '\n' ' '; echo
echo '--- kimi_linear.py kernel refs ---'
grep -rn 'short_conv\|linear_attention\|gdn_attention\|causal_conv1d\|fused_recurrent\|fla_core\|chunk_' "$SITE"/vllm/model_executor/models/kimi_linear.py 2>/dev/null | head -25
echo '--- splitting_ops / layer spec ---'
grep -rn 'splitting_ops\|KimiLinear\|recurrent\|mtp' "$SITE"/vllm/model_executor/models/kimi_linear.py 2>/dev/null | head -20
echo '--- fla / causal_conv1d installed? ---'
for m in fla fla_core causal_conv1d causal_conv1d_cuda; do
  python3 -c "import $m;print('$m',getattr($m,'__version__','?'))" 2>&1 | tail -1
done
echo
echo '--- linear-attention backend 是否可被 --attention-backend 影响 ---'
echo '[kimi_linear.py 里所有 attention/linear 层构造点]'
grep -nE 'Attention\(|LinearAttention|GatedDelta|GDNAttention|ShortConv|kda|KimiDelta|layer_type|KimiLinearDecoderLayer' \
  "$SITE"/vllm/model_executor/models/kimi_linear.py 2>/dev/null | head -30
echo '[gdn_attn.py / short_conv_attn.py 是否注册为可选 backend]'
grep -nE 'class .*Backend|AttentionBackendEnum|register|backend_name' \
  "$SITE"/vllm/v1/attention/backends/gdn_attn.py \
  "$SITE"/vllm/v1/attention/backends/short_conv_attn.py \
  "$SITE"/vllm/v1/attention/backends/linear_attn.py 2>/dev/null | head -30
echo
echo '### Q5 platform capability (no model load)'
python3 - <<'PY' 2>&1 | tail -40
from vllm.platforms import current_platform
import inspect
print('platform class:', type(current_platform).__name__)
for name in ('get_supported_attn_backends','supported_quantization','get_device_capability','is_fp8_supported'):
    v = getattr(current_platform, name, None)
    if v is None:
        print(name, ': MISSING'); continue
    # supported_quantization 是 list 而不是方法，上一版无脑 fn() 把内容吞掉了
    try:
        print(name, '=', v() if callable(v) else v)
    except Exception as e:
        print(name, 'raised', type(e).__name__, e)
PY
} > "$OUT" 2>&1
cat "$OUT"
