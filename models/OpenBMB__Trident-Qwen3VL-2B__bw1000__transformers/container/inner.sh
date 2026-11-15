#!/bin/bash
# 容器内：装依赖 → 起服务 → smoke（文本/图像/图文）→ 分档压测 → 收尾
set -uo pipefail

echo '--- 隔离安装 transformers 4.57.1 + qwen_vl_utils + 匹配的 huggingface_hub ---'
mkdir -p "$PYLIBS"
# 【坑】镜像自带 huggingface_hub 1.6.0，而 transformers 4.57.1 要求 <1.0，导入直接抛 ImportError。
# 用 --no-deps 装进隔离目录，并把 PYTHONPATH 顶到最前面，避免污染镜像。
pip install -q --no-deps --upgrade --target "$PYLIBS" /wheels/huggingface_hub-0.34.0-py3-none-any.whl 2>&1 | tail -1
pip install -q --no-deps --upgrade --target "$PYLIBS" /wheels/transformers-4.57.1-py3-none-any.whl 2>&1 | tail -1
pip install -q --no-deps --upgrade --target "$PYLIBS" /wheels-extra/qwen_vl_utils-0.0.14-py3-none-any.whl 2>&1 | tail -1
export PYTHONPATH="$PYLIBS:${PYTHONPATH:-}"

python3 - <<'PYCHK' || exit 3
import importlib, sys
bad = []
for m in ("torch", "transformers", "qwen_vl_utils", "PIL", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<14} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append(m); print(f"  {m:<14} FAIL {type(e).__name__}: {e}")
import torch
print(f"  device_count   {torch.cuda.device_count()}")
sys.exit(3 if bad else 0)
PYCHK

cd /work/repo
mkdir -p logs bench
echo '--- 起服务（后台） ---'
python3 serve_trident.py --model "$MODEL_PATH" --port "$PORT" \
  --batch-size "$EMB_BATCH" --max-text-tokens "$EMB_MAX_TOKENS" > logs/server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT

echo '--- smoke：文本 / 图像 / 图文三种输入 ---'
python3 - <<'PYSMOKE' || exit 4
import json, sys, time, urllib.request
port = int(__import__("os").environ.get("PORT", "30000"))
t0 = time.time()
while time.time() - t0 < 1800:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            if json.loads(r.read())["status"] == "ok":
                break
    except Exception:
        time.sleep(5)
else:
    print("[smoke] 服务 30 min 未就绪"); sys.exit(1)
print(f"[smoke] 服务就绪 {time.time()-t0:.0f}s")

def post(payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # 把服务端 500 的 body 打出来（里面是真实异常信息），否则只看到一行 HTTPError
        print("[smoke] HTTP", e.code, e.read().decode()[:800]); raise

cases = {
  "text_only": {"input": ["How many stores did Saint Laurent operate in 2020?"], "role": "query"},
  # 用 1024×1024：Qwen3-VL 的 patchify 要求边长能被 16 整除且不小于 min_pixels，
  # 320×240 的小图会触发 `shape ... is invalid for input of size` （实测 2026-11-15）
  "image_only": {"input": [{"image": "/work/repo/assets/red1024.png"}], "role": "doc"},
  "text_image": {"input": [{"text": "a red square", "image": "/work/repo/assets/red1024.png"}], "role": "doc"},
}
import math
for name, p in cases.items():
    d = post(p)
    v = d["data"][0]["embedding"]
    norm = math.sqrt(sum(x*x for x in v))
    print(f"[smoke] {name:<11} dim={len(v)} ||v||={norm:.4f}")
    assert 0.9 < norm < 1.1, "向量未归一化"
print("[smoke] 三种输入全部通过")
PYSMOKE
SMOKE_RC=$?
if [ $SMOKE_RC -ne 0 ]; then echo "[trident] smoke 失败 rc=$SMOKE_RC"; kill $SRV 2>/dev/null; exit 4; fi

echo '--- 分档压测 ---'
python3 bench_embed.py --port "$PORT" --out bench
BENCH_RC=$?

kill $SRV 2>/dev/null
echo "[trident] 服务日志尾部"; tail -5 logs/server.log
echo "[trident] bench rc=$BENCH_RC"
exit $BENCH_RC
