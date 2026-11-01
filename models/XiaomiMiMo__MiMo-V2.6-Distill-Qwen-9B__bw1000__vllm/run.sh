#!/usr/bin/env bash
# 模型启动脚本：装依赖 + 重放 patch + 拉起 vLLM，轮询 /health。
# 容器外由 container/submit.slurm 负责起容器并调用本脚本。
set -euo pipefail
source ./env.sh
mkdir -p logs

if [ -f requirements-extra.txt ]; then
  grep -qvE '^\s*(#|$)' requirements-extra.txt 2>/dev/null && pip install -q --no-deps -r requirements-extra.txt || true
fi
for p in patches/*.patch; do
  [ -e "$p" ] || continue
  if patch -p1 -R -f -s --dry-run -d "$PATCH_ROOT" < "$p" > /dev/null 2>&1; then
    echo "[run] skip $p (already applied)"
  else
    patch -p1 -f -s -d "$PATCH_ROOT" < "$p"
    echo "[run] applied $p"
  fi
done

CMD=(python3 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
     --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP"
     --max-model-len "$MAX_LEN"
     --gpu-memory-utilization "$MEM_FRAC" --max-num-seqs "$MAX_NUM_SEQS"
     --host 127.0.0.1 --port "$PORT")
echo "[run] ${CMD[*]} $EXTRA_ARGS" | tee logs/server.log
"${CMD[@]}" $EXTRA_ARGS >> logs/server.log 2>&1 &
SERVER_PID=$!

for ((t = 0; t < LOAD_TIMEOUT; t += 10)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] server exited, see server.log" >&2; exit 1
  fi
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "[run] server up after ${t}s"
    exit 0
  fi
  sleep 10
done
echo "[run] server not ready in ${LOAD_TIMEOUT}s, see server.log" >&2
kill "$SERVER_PID"; exit 1
