"""压测前置校验（被 bench.sh 调用）。环境变量：PORT / SERVED_NAME。"""
import json, os, sys, urllib.request

port = os.environ.get("PORT", "30000")
name = os.environ.get("SERVED_NAME", os.environ.get("NAME", ""))
base = f"http://127.0.0.1:{port}"
op = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(path, obj, timeout=1800):
    req = urllib.request.Request(base + path, json.dumps(obj).encode(),
                                 {"Content-Type": "application/json"})
    with op.open(req, timeout=timeout) as r:
        return json.load(r)


try:
    with op.open(urllib.request.Request(base + "/v1/models"), timeout=60) as r:
        max_len = json.load(r)["data"][0].get("max_model_len")
except Exception as e:
    print(f"[bench] preflight 失败：无法访问 /v1/models（{e}）")
    sys.exit(1)

print(f"[bench] max_model_len={max_len}")
last = None
for n in (16000, 14000, 12000, 10000, 8000, 4000):
    try:
        d = post("/v1/chat/completions", {"model": name,
                 "messages": [{"role": "user", "content": "海" * n}],
                 "max_tokens": 8, "temperature": 0})
        pt = d.get("usage", {}).get("prompt_tokens")
        print(f"[bench] preflight OK chars={n} prompt_tokens={pt}")
        if pt and pt + 1024 > (max_len or 0):
            print("[bench] WARN: prompt_tokens+1024 仍超 max_model_len")
        elif pt and pt < 8000:
            print("[bench] WARN: 探测长度偏小，16K 档可能未被覆盖")
        sys.exit(0)
    except Exception as e:
        last = f"{type(e).__name__}: {str(e)[:150]}"
        print(f"[bench] chars={n} 失败 -> {last}")
print(f"[bench] preflight 失败（服务端上下文不够或服务异常）：{last}")
print("[bench]         请调大 env.sh 的 MAX_LEN 后重跑，否则 16K 档数据无效")
sys.exit(1)
