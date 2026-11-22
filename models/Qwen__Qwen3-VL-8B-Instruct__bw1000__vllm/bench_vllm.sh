#!/usr/bin/env bash
# 6 档负载（输入 4K/16K × 输出 1K × 并发 1/8/64），每档两轮，取第二轮。
# 用 vLLM 自带的 `vllm bench serve`，**不依赖 SGLang 压测客户端**：
# 2026-11-22 实测本集群 Harbor 不可达、节点 /tmp 也放不下两个镜像（gotcha #62），
# 用同一个镜像里的官方 bench 入口可以省掉 20 GB 和一次镜像 load。
set -uo pipefail
source ./env.sh

mkdir -p bench && rm -f bench/round*.jsonl
echo "[bench] backend=vllm-bench  model=$SERVED_NAME  ctx=$MAX_LEN"

python3 /preflight_lib.py || exit 1

ILENS="${BENCH_ILENS:-4096 16384}"
CONCS="${BENCH_CONCS:-1 8 64}"
ROUNDS="${BENCH_ROUNDS:-1 2}"
if [ "${BENCH_QUICK:-0}" = "1" ]; then
  ILENS=4096; CONCS=1; ROUNDS=1
  echo "[bench] BENCH_QUICK=1：仅 4K/并发1/1轮（自检用）"
fi

for ILEN in $ILENS; do
  for C in $CONCS; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in $ROUNDS; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG $(date +%T)"
      vllm bench serve \
        --backend openai-chat --host 127.0.0.1 --port "$PORT" \
        --endpoint /v1/chat/completions \
        --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
        --dataset-name random --random-input-len "$ILEN" \
        --random-output-len 1024 \
        --max-concurrency "$C" --num-prompts "$N" --seed 1 \
        --save-result --result-dir "bench/raw" --result-filename "${TAG}.json" \
        > "bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG，见 bench/${TAG}.log"
    done
  done
done

# 把 vllm bench 的 json 归并成仓库统一口径的 round1.jsonl / round2.jsonl
python3 - <<'PY'
import json, os, glob
rows = {}
for f in glob.glob("bench/raw/*.json"):
    try:
        d = json.load(open(f, encoding="utf-8"))
    except Exception as e:
        print(f"[merge] skip {f}: {e}"); continue
    base = os.path.basename(f)[:-5]          # round1_4096_1024_c1
    try:
        _, ilen, olen, c = base.split("_")
    except ValueError:
        print(f"[merge] 名字不合规 {base}"); continue
    rnd = int(base.split("_")[0][-1])
    row = {
        "random_input_len": int(ilen), "random_output_len": int(olen),
        "max_concurrency": int(c[1:]),
        "completed": d.get("completed", 0),
        "duration": d.get("duration", 0),
        "request_throughput": d.get("request_throughput", 0),
        "output_throughput": d.get("output_throughput", 0),
        "total_throughput": d.get("total_token_throughput", 0),
        "mean_ttft_ms": d.get("mean_ttft_ms", 0),
        "p99_ttft_ms": d.get("p99_ttft_ms", 0),
        "mean_tpot_ms": d.get("mean_tpot_ms", 0),
        "p99_tpot_ms": d.get("p99_tpot_ms", 0),
        # vllm bench 不直接给 e2e，用 TTFT + TPOT×(output_len-1) 还原：
        # 这是逐请求的确定性关系（无排队时），与 sglang bench 的口径可比
        "mean_e2e_latency_ms": round(d.get("mean_ttft_ms", 0) + d.get("mean_tpot_ms", 0) * (int(olen) - 1), 2),
        "failed": d.get("failed", 0),
    }
    rows.setdefault(rnd, []).append(row)

for rnd, rs in sorted(rows.items()):
    rs.sort(key=lambda r: (r["random_input_len"], r["max_concurrency"]))
    with open(f"bench/round{rnd}.jsonl", "w", encoding="utf-8") as fh:
        for r in rs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[merge] round{rnd}.jsonl  {len(rs)} 行")
PY

# 有效性校验：TTFT=0 或吞吐超物理上限 → 该档无效
echo "[bench] 有效性校验"
python3 - <<'PY'
import json, os, sys
bad = []
for rf in ("bench/round2.jsonl", "bench/round1.jsonl"):
    if not os.path.exists(rf): continue
    for line in open(rf, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        tag = f"{r.get('random_input_len')}_{r.get('max_concurrency')}"
        if not r.get("mean_ttft_ms"):
            bad.append((tag, "TTFT=0（请求疑似被拒）"))
        elif r.get("total_throughput", 0) > 20000:
            bad.append((tag, f"吞吐 {r['total_throughput']:.0f} tok/s 超物理上限"))
    break
if bad:
    for t, why in bad: print(f"  INVALID {t}: {why}")
    sys.exit("  存在无效档位，不要把这些数字写进报告")
print("  all valid")
PY

python3 - <<'EOF' | tee bench/summary.md
import json, os
cols = ["random_input_len","random_output_len","max_concurrency","completed","duration",
        "request_throughput","output_throughput","total_throughput","mean_ttft_ms","p99_ttft_ms",
        "mean_tpot_ms","p99_tpot_ms","mean_e2e_latency_ms"]
print("| " + " | ".join(cols) + " |")
print("|" + " --- |" * len(cols))
rf = "bench/round2.jsonl" if os.path.exists("bench/round2.jsonl") else "bench/round1.jsonl"
try:
    for line in open(rf, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        print("| " + " | ".join(("%.2f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols) + " |")
except FileNotFoundError:
    print("| no data |")
EOF
echo "[bench] done"
