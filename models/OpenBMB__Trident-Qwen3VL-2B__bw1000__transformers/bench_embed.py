#!/usr/bin/env python3
"""Trident embedding 服务的分档压测（只用标准库，客户端侧控并发）。

档位设计：embedding ≠ 生成，没有 output token 的"逐 token 时间"，所以口径换成
    - items/s：每秒完成多少条输入（吞吐，主指标）
    - tok/s：按输入 token 估算的处理速度
    - p50/p95 延迟：单条 e2e（含 HTTP 与 tokenize）

3 个输入长度 × 2 个并发度 = 6 档，和仓库里其它条目的 6 档对齐，便于横向摆在一起看。
每个档位跑 ROUNDS 轮，保留每轮原始数据到 bench/roundN.jsonl，汇总写 bench/summary.md。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TIERS = [(512, 1), (512, 8), (2048, 1), (2048, 8), (4096, 1), (4096, 8)]
ROUNDS = int(os.environ.get("ROUNDS", "2"))
# 每 token 平均 4 个字符（中英混合的经验值），只用于造定长输入与估算 tok/s
CHARS_PER_TOKEN = 4


def make_text(target_tokens: int, idx: int) -> str:
    base = ("The Trident embedding model maps text, images and their combination into one "
            "retrieval space; this sentence exists only to fill a fixed token budget. ")
    unit = max(1, len(base) // CHARS_PER_TOKEN)
    reps = max(1, target_tokens // unit)
    return f"[{idx}] " + base * reps


def post(port: int, payload: dict, timeout: float = 300.0) -> tuple[float, dict]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/embeddings", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return time.time() - t0, data


def wait_health(port: int, limit_s: int = 1800) -> float:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return time.time() - t0
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("服务未就绪")


def run_tier(port: int, tokens: int, conc: int) -> dict:
    texts = [make_text(tokens, i) for i in range(conc)]
    payload = {"input": texts, "role": "doc"}
    lat, resp = post(port, payload)            # 预热（含 CUDA Graph/显存分配）
    dim = len(resp["data"][0]["embedding"])
    rows = []
    for _ in range(ROUNDS):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=conc) as ex:
            # 每个 worker 一条请求（batch=1），并发度由服务端的 GPU 锁串行化 → 测的是排队能力
            futs = [ex.submit(post, port, {"input": [make_text(tokens, i)], "role": "doc"}) for i in range(conc)]
            res = [f.result() for f in futs]
        wall = time.time() - t0
        lat_s = [r[0] for r in res]
        rows.append({
            "tokens": tokens, "concurrency": conc, "items": conc, "wall_s": round(wall, 3),
            "items_per_s": round(conc / wall, 3),
            "tokens_per_s": round(conc * tokens / wall, 1),
            "lat_p50_ms": round(statistics.median(lat_s) * 1000, 1),
            "lat_p95_ms": round(sorted(lat_s)[max(0, int(len(lat_s) * 0.95) - 1)] * 1000, 1),
        })
    first = rows[0]
    return {"tokens": tokens, "concurrency": conc, "dim": dim,
            "items_per_s": round(statistics.mean([r["items_per_s"] for r in rows]), 3),
            "tokens_per_s": round(statistics.mean([r["tokens_per_s"] for r in rows]), 1),
            "lat_p50_ms": first["lat_p50_ms"], "lat_p95_ms": first["lat_p95_ms"],
            "rounds": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "30000")))
    ap.add_argument("--out", default="bench")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    boot = wait_health(args.port)
    print(f"[bench] 服务就绪（{boot:.1f}s）", flush=True)

    results = []
    for tokens, conc in TIERS:
        r = run_tier(args.port, tokens, conc)
        results.append(r)
        print(f"[bench] {tokens} tok × c{conc}: {r['items_per_s']} items/s, "
              f"{r['tokens_per_s']} tok/s, p50 {r['lat_p50_ms']} ms", flush=True)

    with open(os.path.join(args.out, "rounds.jsonl"), "w", encoding="utf-8") as f:
        for r in results:
            for row in r["rounds"]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = ["| 输入长度 | 并发 | items/s | tok/s | p50 (ms) | p95 (ms) | 向量维度 |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in results:
        lines.append(f"| {r['tokens']} | {r['concurrency']} | {r['items_per_s']} | "
                     f"{r['tokens_per_s']} | {r['lat_p50_ms']} | {r['lat_p95_ms']} | {r['dim']} |")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Trident-Qwen3VL-2B 分档压测（BW1000）\n\n")
        f.write(f"轮数 {ROUNDS}/档，取均值；口径见 bench_embed.py 顶部注释。\n\n")
        f.write("\n".join(lines) + "\n")
    print("[bench] 已写 bench/summary.md", flush=True)


if __name__ == "__main__":
    main()
