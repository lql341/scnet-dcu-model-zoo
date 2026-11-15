#!/usr/bin/env bash
# 起服务：trident_embedder（Transformers 直载）+ OpenAI 兼容 HTTP。
# 在【容器内】执行；容器外由 container/submit.slurm 负责起容器。
set -euo pipefail
source ./env.sh
mkdir -p logs
export PYTHONPATH="${PYLIBS:-/opt/trident-libs}:${PYTHONPATH:-}"
exec python3 serve_trident.py --model "$MODEL_PATH" --port "$PORT" \
  --batch-size "${EMB_BATCH:-16}" --max-text-tokens "${EMB_MAX_TOKENS:-4096}"
