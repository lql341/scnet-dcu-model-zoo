#!/usr/bin/env bash
# 模型启动脚本：装依赖 + 重放 patch + 拉起 SGLang，轮询 /health。
# 在【容器内】执行；容器外由 container/submit.slurm 负责起容器并调用本脚本。
set -euo pipefail
source ./env.sh
mkdir -p logs

# 1) 适配改动（本件无包升级、无 patch，两段自动跳过）
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

# 2) 拼启动命令
# vLLM 用的参数名与 SGLang 不同：--max-model-len / --gpu-memory-utilization / --max-num-seqs。
# 早期版本这里是从 SGLang 条目复制来的，引用了本件并未定义的 $MAX_RUNNING_REQUESTS，
# 在 set -u 下会直接 unbound variable。2026-11-15 修正。
CMD=(python3 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
     --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN"
     --gpu-memory-utilization "$MEM_FRAC"
     --max-num-seqs "$MAX_NUM_SEQS"
     --host 127.0.0.1 --port "$PORT")
echo "[run] ${CMD[*]} $EXTRA_ARGS" | tee logs/server.log
"${CMD[@]}" $EXTRA_ARGS >> logs/server.log 2>&1 &
SERVER_PID=$!

# 3) 轮询 /health
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
