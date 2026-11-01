#!/usr/bin/env bash
# 性能测试：6 档负载（输入 4K/16K × 输出 1K × 并发 1/8/64），每档连测两次，第一次预热，取第二次。
# 首次提交的结果即该「模型 × 卡型 × 框架 × 精度」的性能基准。
set -uo pipefail   # 不加 -e：某一档失败时记录下来，继续跑其余档
source ./env.sh

BENCH_MODE=${BENCH_MODE:-text}
mkdir -p bench && rm -f bench/round*.jsonl   # 结果文件是追加写入，重跑前清掉旧结果

BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving   # 旧版 sglang 的模块名
echo "[bench] module=$BENCH  mode=$BENCH_MODE  ctx=$MAX_LEN"
# 说明：vLLM 镜像不含 sglang 压测客户端。官方口径要求用 sglang 的 bench 客户端
#       （「SGLang / vLLM 服务通用」），因此客户端从 CLI_IMAGE 容器里跑，
#       vLLM 服务容器用 --network=host 暴露到节点 127.0.0.1:$PORT。见 container/submit.slurm

# 前置校验：确认服务端容得下 16K 输入 + 1K 输出。
# 注意：不使用 /tokenize —— sglang 在该类模型（词表 id 较大）上会用 orjson 序列化
# token id 并抛 TypeError: Integer exceeds 64-bit range，接口返回 500。
# 这里用「长度阶梯 + 真实 chat 请求」探测，任一长度成功即可。
python3 /preflight_lib.py || exit 1

DATA=(--dataset-name random-ids)   # 随机 token，无需下载数据集
# 档位集合可覆盖：用于复现自检（BENCH_QUICK=1 只跑 4K/并发1/1轮），不影响正式口径
ILENS="${BENCH_ILENS:-4096 16384}"
CONCS="${BENCH_CONCS:-1 8 64}"
ROUNDS="${BENCH_ROUNDS:-1 2}"
if [ "${BENCH_QUICK:-0}" = "1" ]; then ILENS=4096; CONCS=1; ROUNDS=1; echo "[bench] BENCH_QUICK=1：仅 4K/并发1/1轮，用于复现自检"; fi
for ILEN in $ILENS; do
  for C in $CONCS; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))   # 请求数：并发的 2 倍，至少 8 个
    for R in $ROUNDS; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG $(date +%T)"
      python3 -m "$BENCH" \
        --backend "$FRAMEWORK" --host 127.0.0.1 --port "$PORT" \
        --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" "${DATA[@]}" \
        --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency "$C" --num-prompts "$N" --seed 1 \
        --output-file "bench/round${R}.jsonl" > "bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG，见 bench/${TAG}.log；不支持的档位写进报告「注意事项」"
    done
  done
done

# 有效性校验：TTFT=0 或吞吐超物理上限，说明请求被拒但被计为完成
echo "[bench] 有效性校验"
python3 - <<'PY'
import json, sys
bad = []
ROUND_FILE = "bench/round2.jsonl" if os.path.exists("bench/round2.jsonl") else "bench/round1.jsonl"
EXPECT = len(ILENS.split()) * len(CONCS.split())
for line in open(ROUND_FILE, encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    tag = f"{r.get('random_input_len')}_{r.get('max_concurrency')}"
    if not r.get("mean_ttft_ms"):
        bad.append((tag, "TTFT=0（请求疑似被拒）"))
    elif r.get("total_throughput", 0) > 20000:
        bad.append((tag, f"吞吐 {r['total_throughput']:.0f} tok/s 超物理上限"))
if bad:
    for t, why in bad:
        print(f"  INVALID {t}: {why}")
    sys.exit("  存在无效档位，请调大 MAX_LEN 后重跑；不要把这些数字写进报告")
print("  all valid")
PY

python3 - <<'EOF' | tee bench/summary.md
import json, os
ILENS = os.environ.get("BENCH_ILENS", "4096 16384")
CONCS = os.environ.get("BENCH_CONCS", "1 8 64")
if os.environ.get("BENCH_QUICK") == "1":
    ILENS, CONCS = "4096", "1"
EXPECT = len(ILENS.split()) * len(CONCS.split())
cols = ["random_input_len", "random_output_len", "max_concurrency", "completed", "duration",
        "request_throughput", "output_throughput", "total_throughput", "mean_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms", "p99_tpot_ms", "mean_e2e_latency_ms"]
print("| " + " | ".join(cols) + " |")
print("|" + " --- |" * len(cols))
round_file = "bench/round2.jsonl" if os.path.exists("bench/round2.jsonl") else "bench/round1.jsonl"
try:
    for line in open(round_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        print("| " + " | ".join(("%.2f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c, "-")) for c in cols) + " |")
except FileNotFoundError:
    print("| no data |")
EOF
echo "[bench] done"
