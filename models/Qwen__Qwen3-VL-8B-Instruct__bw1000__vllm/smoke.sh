#!/usr/bin/env bash
# 多模态 smoke：文本 + 单图 + 多图（T02 要求「图像预处理 → vision encoder → LLM 全链路，支持多图」）
set -euo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS -f --noproxy "*" --max-time 600)
exec > >(tee logs/smoke.log) 2>&1

echo "== /health";    "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models"; "${CURL[@]}" "$URL/v1/models"; echo

echo "== 纯文本"
"${CURL[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话说明什么是空间推理。\"}],\"temperature\":0,\"max_tokens\":128}"
echo

# 造两张小图（纯 Python，不依赖外部素材）
python3 - <<'PY'
import struct, zlib, os
def png(path, w, h, rgb):
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    data = (b'\x89PNG\r\n\x1a\n'
            + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(raw))
            + chunk(b'IEND', b''))
    open(path, 'wb').write(data)
os.makedirs('assets', exist_ok=True)
png('assets/red.png', 320, 240, (220, 40, 40))
png('assets/blue.png', 320, 240, (40, 40, 220))
print('生成 assets/red.png (320x240) 与 assets/blue.png (320x240)')
PY
B64A=$(base64 -w0 assets/red.png)
B64B=$(base64 -w0 assets/blue.png)

echo "== 单图"
"${CURL[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$SERVED_NAME\",
  \"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64A\"}},
    {\"type\":\"text\",\"text\":\"这张图是什么颜色？只回答颜色。\"}]}],
  \"temperature\":0,\"max_tokens\":64}"
echo

echo "== 多图（2 张）"
"${CURL[@]}" "$URL/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\":\"$SERVED_NAME\",
  \"messages\":[{\"role\":\"user\",\"content\":[
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64A\"}},
    {\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64B\"}},
    {\"type\":\"text\",\"text\":\"这两张图分别是什么颜色？\"}]}],
  \"temperature\":0,\"max_tokens\":128}"
echo
