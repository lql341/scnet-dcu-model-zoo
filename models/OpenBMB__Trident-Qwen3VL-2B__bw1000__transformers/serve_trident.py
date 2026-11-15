#!/usr/bin/env python3
"""OpenAI 兼容的 Trident embedding 服务（最小实现，只用标准库 + torch）。

为什么不用 FastAPI：镜像里不保证有；这个服务只需要两个端点，标准库的
ThreadingHTTPServer 足够，且省掉一层依赖风险。

端点
    GET  /health           → {"status":"ok","model":...}
    GET  /v1/models        → OpenAI 风格的模型列表
    POST /v1/embeddings    → {"input": str | [str] | [{"text":..,"image":..}], "role": "query"|"doc"}

与 OpenAI 的差异（在 README 里写明）：`role` 是 Trident 特有参数，缺省 query；
返回 `data[i].embedding` 为 L2 归一化后的 float 列表，`usage.prompt_tokens` 为估算值。

并发模型：GPU 侧串行（一把锁），HTTP 侧多线程。对 embedding 服务这是最稳的选择：
一次 forward 已经吃满算力，多请求并发只会争显存；真正要提吞吐应该增大 batch，
由 bench_embed.py 在客户端控制并发度来测。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import traceback
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trident_embedder import TridentEmbedder  # noqa: E402

MODEL: TridentEmbedder | None = None
LOCK = threading.Lock()
STATS = {"requests": 0, "items": 0, "seconds": 0.0}


def _items(payload: dict) -> tuple[list, str]:
    inp = payload.get("input")
    role = payload.get("role") or "query"
    if isinstance(inp, str):
        return [{"text": inp}], role
    if isinstance(inp, list) and inp and all(isinstance(x, str) for x in inp):
        return [{"text": s} for s in inp], role
    if isinstance(inp, list) and inp and all(isinstance(x, dict) for x in inp):
        return inp, role
    raise ValueError("input 必须是 str、[str] 或 [{text/image}]")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # 静音：bench 自己记日志
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok" if MODEL is not None else "loading",
                             "model": os.environ.get("MODEL_PATH", "?"),
                             "stats": STATS})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": os.environ.get("MODEL_ID", "trident"), "object": "model",
                 "owned_by": "OpenBMB"}]})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self._send(404, {"error": {"message": "not found"}})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
            items, role = _items(payload)
        except Exception as e:
            self._send(400, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        t0 = time.time()
        try:
            with LOCK:                       # GPU 串行，避免显存竞争
                vecs = MODEL.process(items, role=role)
        except Exception as e:
            # 服务端也要把栈打到 server.log：否则客户端只看到 500，看不出缺哪个依赖
            traceback.print_exc()
            self._send(500, {"error": {"message": f"{type(e).__name__}: {e}"}})
            return
        dt = time.time() - t0
        STATS["requests"] += 1
        STATS["items"] += len(items)
        STATS["seconds"] += dt
        data = [{"object": "embedding", "index": i,
                 "embedding": vecs[i].tolist()} for i in range(len(vecs))]
        self._send(200, {
            "object": "list", "data": data,
            "model": os.environ.get("MODEL_ID", "trident"),
            "usage": {"prompt_tokens": payload.get("_tokens", 0),
                      "total_tokens": payload.get("_tokens", 0)},
        })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL_PATH", "OpenBMB/Trident-Qwen3VL-2B"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "30000")))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--batch-size", type=int, default=int(os.environ.get("EMB_BATCH", "16")))
    ap.add_argument("--max-text-tokens", type=int, default=int(os.environ.get("EMB_MAX_TOKENS", "4096")))
    args = ap.parse_args()

    global MODEL
    t0 = time.time()
    MODEL = TridentEmbedder(args.model, batch_size=args.batch_size,
                            max_text_tokens=args.max_text_tokens)
    print(f"[serve] 模型加载完成 {time.time() - t0:.1f}s，监听 {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
