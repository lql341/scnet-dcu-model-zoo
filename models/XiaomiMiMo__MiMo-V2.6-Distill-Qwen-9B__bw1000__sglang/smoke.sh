#!/usr/bin/env bash
# Smoke 测试：健康检查 + 模型列表 + 单轮 + 多轮 + 流式；输出写入 smoke.log。
# 任一接口非 2xx 即失败。
set -euo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS -f --noproxy "*" --max-time 300)
exec > >(tee logs/smoke.log) 2>&1

chat() {  # $1 = messages（JSON 数组），$2 = 是否流式
  "${CURL[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$SERVED_NAME\",\"messages\":$1,\"temperature\":0,\"max_tokens\":256,\"stream\":$2}"
  echo
}

echo "== /health";    "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models"; "${CURL[@]}" "$URL/v1/models"; echo
echo "== 单轮"; chat '[{"role":"user","content":"用一句话介绍国产异构加速器。"}]' false
echo "== 多轮"; chat '[{"role":"user","content":"我叫小海。"},{"role":"assistant","content":"你好，小海！"},{"role":"user","content":"我叫什么名字？"}]' false
echo "== 流式"; chat '[{"role":"user","content":"用一句话介绍国产异构加速器。"}]' true | tail -n 3
