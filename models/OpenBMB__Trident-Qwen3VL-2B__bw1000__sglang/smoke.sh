#!/usr/bin/env bash
# 嵌入模型 smoke：纯文本 / 单图 / 图文融合 三条向量化路径 + 余弦相似度排序校验
# 与 chat 模型不同，接口是 /v1/embeddings
set -euo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS -f --noproxy "*" --max-time 600)
exec > >(tee logs/smoke.log) 2>&1

echo "== /health";    "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models"; "${CURL[@]}" "$URL/v1/models"; echo

python3 - <<'PY'
import struct, zlib, os
def png(path, w, h, rgb):
    raw = b''.join(b'\x00' + bytes(rgb) * w for _ in range(h))
    def chunk(t, d):
        c = t + d
        return struct.pack('>I', len(d)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)
    open(path, 'wb').write(b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))
os.makedirs('assets', exist_ok=True)
png('assets/red.png', 320, 240, (220, 40, 40))
png('assets/blue.png', 320, 240, (40, 40, 220))
print('生成 assets/red.png 与 assets/blue.png')
PY

echo "== 纯文本嵌入"
"${CURL[@]}" "$URL/v1/embeddings" -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"input\":\"一只红色的苹果\"}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
e=d['data'][0]['embedding']
print('  维度:', len(e), '| 前 5 个值:', [round(x,4) for x in e[:5]], '| usage:', d.get('usage'))
"

echo "== 单图嵌入"
B64=$(base64 -w0 assets/red.png)
"${CURL[@]}" "$URL/v1/embeddings" -H "Content-Type: application/json" -d "{
  \"model\":\"$SERVED_NAME\",
  \"input\":[{\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$B64\"}}]}]}" \
  | python3 -c "
import json,sys
d=json.load(sys.stdin)
e=d['data'][0]['embedding']
print('  维度:', len(e), '| 前 5 个值:', [round(x,4) for x in e[:5]])
"

echo "== 图文融合嵌入 + 相似度排序"
python3 - <<'PY'
import base64, json, urllib.request
PORT=__import__('os').environ.get('PORT','30000')
NAME=__import__('os').environ.get('SERVED_NAME','trident-qwen3vl-2b')
red=base64.b64encode(open('assets/red.png','rb').read()).decode()
blue=base64.b64encode(open('assets/blue.png','rb').read()).decode()
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def emb(payload):
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings",
        json.dumps({"model":NAME,"input":payload}).encode(),{"Content-Type":"application/json"})
    with op.open(r,timeout=600) as f:
        return json.load(f)["data"][0]["embedding"]
def cos(a,b):
    import math
    na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b))/(na*nb) if na and nb else 0.0
q = emb("红色的物体")
img_red = emb([{"content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{red}"}}]}])
img_blue = emb([{"content":[{"type":"image_url","image_url":{"url":f"data:image/png;base64,{blue}"}}]}])
s_red, s_blue = cos(q,img_red), cos(q,img_blue)
print(f"  query=「红色的物体」  vs 红图 cos={s_red:.4f}  vs 蓝图 cos={s_blue:.4f}")
print(f"  排序: {'红图 > 蓝图 ✅' if s_red > s_blue else '红图 ≤ 蓝图 ❌（检索方向反了）'}")
PY
