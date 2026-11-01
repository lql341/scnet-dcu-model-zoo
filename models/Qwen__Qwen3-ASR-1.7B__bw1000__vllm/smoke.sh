#!/usr/bin/env bash
# ASR smoke：用 macOS TTS 生成的已知转写音频做识别，校验识别结果
set -euo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS -f --noproxy "*" --max-time 300)
exec > >(tee logs/smoke.log) 2>&1

echo "== /health";    "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models"; "${CURL[@]}" "$URL/v1/models"; echo

echo "== 音频信息"; ls -l "$AUDIO"

echo "== audio_url（chat/completions）"
# 不能把 base64 内联进 -d：103 KB 音频编码后约 138 KB，会触发 Argument list too long
python3 - > /tmp/asr_req.json <<'PY'
import base64, json, os
audio = os.environ["AUDIO"]
b64 = base64.b64encode(open(audio, "rb").read()).decode()
print(json.dumps({
    "model": os.environ["SERVED_NAME"],
    "messages": [{"role": "user", "content": [
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}]}],
    "temperature": 0, "max_tokens": 256}))
PY
"${CURL[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" --data @/tmp/asr_req.json
echo

echo "== transcriptions 接口"
"${CURL[@]}" "$URL/v1/audio/transcriptions" \
  -F "file=@$AUDIO" -F "model=$SERVED_NAME"
echo

echo "== 参考文本"; echo "$AUDIO_TEXT"
