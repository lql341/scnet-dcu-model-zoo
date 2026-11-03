#!/bin/bash
# Qwen-Image 容器内步骤（独立脚本，避免嵌套引号）
set -uo pipefail
LIBS='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'
export LD_LIBRARY_PATH="$LIBS:${LD_LIBRARY_PATH:-}"
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo '--- 镜像内版本 ---'
python3 -c 'import torch, transformers; print("torch", torch.__version__, "| transformers", transformers.__version__)'

echo '--- 隔离安装 diffusers 0.41.0 + huggingface_hub 1.33.0 ---'
# 【2026-11-03 修正】diffusers 0.41.0 的元数据要求 huggingface-hub>=1.32.0,<2.0。
# 之前装的是 hub 0.36.2（且 --no-deps 不会报冲突），于是 diffusers 一导入就缺
# `resolve_revision` / `RevisionResolutionError`。上游并没有“未来版本 API”——
# hub 1.32.0 / 1.33.0 同时导出这两个符号（已逐个轮子验证），装上即可，不再需要 shim。
pip install -q --no-deps --target /opt/qi-libs /wheels/diffusers-0.41.0-py3-none-any.whl 2>&1 | tail -2
pip install -q --no-deps --target /opt/qi-libs /wheels/huggingface_hub-1.33.0-py3-none-any.whl 2>&1 | tail -2

export PYTHONPATH=/opt/qi-libs:${PYTHONPATH:-}
python3 - <<'PY'
import huggingface_hub as hub
print('hub', hub.__version__, '| resolve_revision', hasattr(hub, 'resolve_revision'))
from huggingface_hub.errors import RevisionResolutionError  # noqa: F401
import diffusers
print('diffusers', diffusers.__version__)
from diffusers import QwenImage21Pipeline
print('QwenImage21Pipeline OK')
PY

echo '--- 官方条件评测 ---'
cd /work/repo
python3 evaluate_official.py 2>&1 | tail -30
