#!/usr/bin/env bash
set -euo pipefail
source ./env.sh
mkdir -p logs
CMD=(python3 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
     --host 127.0.0.1 --port "$PORT"
     --tensor-parallel-size "$TP"
     --max-model-len "$MAX_LEN"
     --gpu-memory-utilization "$MEM_FRAC")
echo "[run] ${CMD[*]} $EXTRA_ARGS" | tee logs/server.log
"${CMD[@]}" $EXTRA_ARGS >> logs/server.log 2>&1 &
SERVER_PID=$!
for ((t = 0; t < LOAD_TIMEOUT; t += 10)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[run] server exited" >&2; exit 1; fi
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "[run] server up after ${t}s"; exit 0
  fi
  sleep 10
done
echo "[run] timeout" >&2; kill "$SERVER_PID"; exit 1
