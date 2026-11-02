#!/usr/bin/env bash
# 嵌入模型 smoke（vLLM）：纯文本 / 单图 / 图文相似度排序
set -euo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS -f --noproxy "*" --max-time 600)
exec > >(tee logs/smoke.log) 2>&1
echo "== /health";    "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models"; "${CURL[@]}" "$URL/v1/models"; echo
echo "== 纯文本嵌入"
"${CURL[@]}" "$URL/v1/embeddings" -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"input\":\"一只红色的苹果\"}" | head -c 300; echo
