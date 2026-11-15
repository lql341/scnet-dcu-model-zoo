#!/bin/bash
# 容器内只读探测：回答 16K/c64 下一枪该打哪里。不加载模型、不起服务。
#   ① 0.20 的 RocmPlatform 里 MLA 到底允许哪些 backend（日志只说 potential=['TRITON_MLA']）
#   ② --attention-backend FLASHMLA 在 0.20 上是否会被接受
#   ③ 0.21 的 vllm_hcu 加载 3D 权重时无条件 .t()，有没有开关能绕开
# 由 kimi-backend-probe-2026-11-15.slurm 挂到 /gate/probe.sh 调用
set -uo pipefail
TAG="${1:?usage: probe.sh <tag>}"
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/$TAG.md

{
echo "##### $TAG #####"
echo
echo '### A. MLA backend 的判定逻辑（0.20 关键）'
python3 - <<'PY' 2>&1 | tail -60
import inspect
from vllm.platforms import rocm
for name in ('get_attn_backend_cls', 'get_valid_backends', '_find_non_ssm_backend', 'get_supported_vit_attn_backends'):
    fn = getattr(rocm.RocmPlatform, name, None)
    if fn is None:
        print(name, 'MISSING'); continue
    print('='*20, name, '='*20)
    try:
        src = inspect.getsource(fn)
        print(src[:3000])
    except Exception as e:
        print('  source unavailable:', type(e).__name__, e)
PY
echo
echo '### A2. rocm.py 里出现 MLA / FLASHMLA 的行'
grep -nE 'FLASHMLA|TRITON_MLA|AITER_MLA|MLA_SPARSE|use_aiter|aiter' "$SITE"/vllm/platforms/rocm.py 2>/dev/null | head -40
echo
echo '### A3. env.py 里与 attention backend 相关的开关'
grep -nE 'ATTENTION_BACKEND|FLASHMLA|MLA' "$SITE"/vllm/envs.py 2>/dev/null | head -20
echo
echo '### A4. MLA backend 的可用性查询（不加载模型，只问枚举）'
python3 - <<'PY' 2>&1 | tail -30
try:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
    print('MLA 相关枚举:', [e.name for e in AttentionBackendEnum if 'MLA' in e.name])
    for e in AttentionBackendEnum:
        if 'MLA' in e.name:
            try:
                cls = e.get_class()
                print(f'  {e.name} -> {cls.__module__}.{cls.__name__}  cg_support={getattr(cls, "get_cudagraph_support", lambda *a: "?")()}')
            except Exception as ex:
                print(f'  {e.name} -> 无法实例化 {type(ex).__name__}: {str(ex)[:120]}')
except Exception as e:
    print('registry failed:', type(e).__name__, e)
PY
echo
echo '### B. 0.21 vllm_hcu 的 3D 权重加载路径'
if [ -d "$SITE"/vllm_hcu ]; then
  echo '--- parameter.py:120-200 ---'
  sed -n '120,200p' "$SITE"/vllm_hcu/model_executor/parameter.py 2>/dev/null
  echo '--- parameter.py 里所有 .t() 调用点 ---'
  grep -nE '\.t\(\)|transpose|is_quantization|def load_' "$SITE"/vllm_hcu/model_executor/parameter.py 2>/dev/null | head -40
  echo '--- linear.py:600-650 ---'
  sed -n '600,650p' "$SITE"/vllm_hcu/model_executor/layers/linear.py 2>/dev/null
  echo '--- vllm_hcu 里是否有 env 开关可绕开 ---'
  grep -rnE 'environ|getenv|VLLM_HCU_[A-Z_]+' "$SITE"/vllm_hcu/model_executor/parameter.py 2>/dev/null | head -20
else
  echo '（本镜像无 vllm_hcu）'
fi
echo
echo '### C. Kimi 的 conv1d 权重形状（判断 3D 从哪来）'
grep -nE 'conv1d|short_conv|weight_loader|unsqueeze' "$SITE"/vllm/model_executor/models/kimi_linear.py 2>/dev/null | head -25
echo
echo '### D. 本地 Kimi 权重里 conv1d 的实际 shape'
python3 - <<'PY' 2>&1 | tail -12
import json, os
p = '/models/Kimi/model.safetensors.index.json'
if not os.path.exists(p):
    print('无 index.json'); raise SystemExit
idx = json.load(open(p))['weight_map']
c = [(k, v) for k, v in idx.items() if 'conv1d' in k][:5]
print('conv1d 张量样例:', c)
PY
} > "$OUT" 2>&1
cat "$OUT"
