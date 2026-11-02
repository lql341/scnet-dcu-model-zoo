#!/bin/bash
# Qwen-Image 容器内步骤（独立脚本，避免嵌套引号）
set -uo pipefail
LIBS='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo '--- 镜像内版本 ---'
python3 -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'

echo '--- 隔离安装 diffusers 0.41.0 + hub 0.36.2 ---'
pip install -q --no-deps --target /opt/qi-libs /wheels/diffusers-0.41.0-py3-none-any.whl 2>&1 | tail -2
pip install -q --no-deps --target /opt/qi-libs /wheels/huggingface_hub-0.36.2-py3-none-any.whl 2>&1 | tail -2

# diffusers 0.41.0 使用 `from huggingface_hub import resolve_revision`，
# 而实测 hub 0.24.7/0.26.2/0.30.0/0.33.0/0.36.2/1.11.0 均无该符号（上游“未来版本 API”）。
# 本项目只用本地权重，revision 无需解析，故以原样返回等价实现。
cat > /opt/qi-libs/hub_shim.py <<'SHIM'
import huggingface_hub as _hub

if not hasattr(_hub, "resolve_revision"):
    def resolve_revision(repo_id=None, revision=None, **kwargs):
        return revision
    _hub.resolve_revision = resolve_revision
SHIM

export PYTHONPATH=/opt/qi-libs:${PYTHONPATH:-}
python3 -c 'import hub_shim, diffusers; print("diffusers", diffusers.__version__)'
python3 -c 'from diffusers import QwenImage21Pipeline; print("QwenImage21Pipeline OK")' 2>&1 | tail -6

echo '--- 官方条件评测 ---'
cd /work/repo
python3 evaluate_official.py 2>&1 | tail -30
