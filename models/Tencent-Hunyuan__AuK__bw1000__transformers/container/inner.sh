#!/bin/bash
# AuK 容器内步骤（独立脚本，避免在 podman run -c "..." 里嵌套引号）
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo '--- 镜像内版本 ---'
python3 -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'

echo '--- 逐个安装 wheel（单个失败不中断）---'
n_ok=0; n_fail=0
for w in /wheels/*.whl; do
  if pip install -q --no-deps --target /opt/auk-libs "$w" 2>/dev/null; then
    n_ok=$((n_ok + 1))
  else
    n_fail=$((n_fail + 1)); echo "  [warn] fail: $(basename "$w")"
  fi
done
echo "已装 $n_ok 个，失败 $n_fail 个"

export PYTHONPATH=/opt/auk-libs:/auk-src/src:${PYTHONPATH:-}
echo '--- 依赖可用性校验（cp310 轮子）---'
python3 - <<'PY'
import importlib, sys
ok = True
for m in ("numpy", "scipy", "numba", "llvmlite", "soundfile", "soxr", "librosa", "joblib", "sklearn"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<10} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        ok = False
        print(f"  {m:<10} FAIL {type(e).__name__}: {e}")
sys.exit(0 if ok else 3)
PY
[ $? -eq 3 ] && { echo '[inner] 依赖不全，后续评测无意义，提前退出'; exit 3; }

echo '--- 导入测试 ---'
python3 -c 'import auk; print("auk import OK")' 2>&1 | tail -5
python3 -c 'from auk.infer.infer_cli import main; print("infer_cli OK")' 2>&1 | tail -8
echo '--- 官方条件评测（PROBE_ONLY 可限制条数）---'
cd /work/repo
export MODEL_PATH='/models/AuK' ENCODER_PATH='/models/Qwen2.5-Omni-3B'
export COND_FILE=conditions/auk.jsonl ASSET_ROOT=/assets OUTDIR=outputs
mkdir -p outputs
python3 evaluate_official.py 2>&1 | tail -35
