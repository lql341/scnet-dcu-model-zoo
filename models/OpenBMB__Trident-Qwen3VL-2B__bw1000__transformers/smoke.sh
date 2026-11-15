#!/usr/bin/env bash
# 嵌入模型 smoke：纯文本 / 单图 / 图文融合三条路径 + 向量 L2 归一化校验。
# 与 chat 模型不同，主接口是 /v1/embeddings。
set -uo pipefail
source ./env.sh
mkdir -p logs
URL="http://127.0.0.1:${PORT:-30000}"
exec > >(tee logs/smoke.log) 2>&1

echo "== /health";     curl -sS -f --noproxy '*' --max-time 60 "$URL/health"; echo
echo "== /v1/models";  curl -sS -f --noproxy '*' --max-time 60 "$URL/v1/models"; echo
echo "== 三条向量化路径 =="
python3 - "$URL" <<'PY'
import json, math, sys, urllib.request
url = sys.argv[1]
def post(p):
    req = urllib.request.Request(url + "/v1/embeddings", data=json.dumps(p).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())
cases = {
  "text_only":  {"input": ["How many stores did Saint Laurent operate in 2020?"], "role": "query"},
  "image_only": {"input": [{"image": "assets/red1024.png"}], "role": "doc"},
  "text_image": {"input": [{"text": "a red square", "image": "assets/red1024.png"}], "role": "doc"},
}
for k, p in cases.items():
    v = post(p)["data"][0]["embedding"]
    n = math.sqrt(sum(x * x for x in v))
    print(f"{k:<11} dim={len(v)} ||v||={n:.4f}")
    assert 0.9 < n < 1.1, "向量未 L2 归一化"
print("三条路径全部通过")
PY
