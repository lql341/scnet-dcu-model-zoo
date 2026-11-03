#!/bin/bash -l
# BW1000 × vLLM × MiMo：起服务 → preflight → smoke → bench 6档×2 → summary + 有效性校验
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J hrvbench
#SBATCH -o /public/home/scnethr2601a10/work/vllm/bench-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vllm/bench-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vllm
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
# 每作业独立 root：避免同节点上一个作业残留的 podman pause/storage 状态导致 pull 直接失败
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }
echo "== pull =="
S=$(date +%s); $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1
echo "pull_rc=$? $(( $(date +%s) - S ))s"
$PD image exists "$IMG" || { grep -iE "^Error" /tmp/pull.log|tail -3; exit 1; }

cat > /tmp/inner.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000; CTX=20480
export NAME PORT

echo "=== start vllm ctx=$CTX $(date +%T) ==="
cd /root
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name "$NAME" \
  --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len $CTX --gpu-memory-utilization 0.97 --enforce-eager \
  > "$W/server-bench.log" 2>&1 &
SPID=$!
OK=0
for t in $(seq 1 180); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_EXITED at $((t*10))s"; break; }
  curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { OK=1; echo "SERVER_UP after $((t*10))s $(date +%T)"; break; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "SERVER_NOT_UP"; grep -iE "error|No available memory|Traceback" "$W/server-bench.log"|tail -8; tail -12 "$W/server-bench.log"; exit 1; }

echo "=== preflight：用 /tokenize 精确定长，确认服务端容得下 16K+1K ==="
python3 /preflight.py || { echo "PREFLIGHT_FAIL"; exit 1; }

echo "=== bench $(date +%T) ==="
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
echo "bench_module=$BENCH"
mkdir -p "$W/bench"; rm -f "$W"/bench/round*.jsonl
for ILEN in 4096 16384; do
  for C in 1 8 64; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG $(date +%T)"
      timeout 2400 python3 -m "$BENCH" \
        --backend vllm --host 127.0.0.1 --port $PORT \
        --model "$NAME" --tokenizer "$MODEL" \
        --dataset-name random-ids \
        --random-input-len $ILEN --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency $C --num-prompts $N --seed 1 \
        --output-file "$W/bench/round${R}.jsonl" > "$W/bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG"
    done
  done
done

echo "=== summary $(date +%T) ==="
python3 - <<'PY' > "$W/bench/summary.md" 2>&1
import json, sys
cols = ["random_input_len","random_output_len","max_concurrency","completed","duration",
        "request_throughput","output_throughput","total_throughput","mean_ttft_ms","p99_ttft_ms",
        "mean_tpot_ms","p99_tpot_ms","mean_e2e_latency_ms"]
print("| " + " | ".join(cols) + " |")
print("|" + " --- |"*len(cols))
rows=[]
for line in open("/work/bench/round2.jsonl", encoding="utf-8"):
    line=line.strip()
    if not line: continue
    r=json.loads(line); rows.append(r)
    print("| " + " | ".join(("%.2f"%r[c]) if isinstance(r.get(c),float) else str(r.get(c,"-")) for c in cols) + " |")
bad=[]
for r in rows:
    tag="%s_%s"%(r.get("random_input_len"),r.get("max_concurrency"))
    if not r.get("mean_ttft_ms"): bad.append((tag,"TTFT=0"))
    elif r.get("total_throughput",0)>20000: bad.append((tag,"throughput too high"))
print()
print("### 有效性校验")
print("all valid" if not bad and len(rows)==6 else "INVALID: %s" % bad)
PY
cat "$W/bench/summary.md"
pkill -9 -f "entrypoints.openai.api_server" 2>/dev/null || true
echo "=== DONE inner $(date +%T) ==="
EOS

echo "== run $(date +%T) =="
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  -v /tmp/inner.sh:/inner.sh:ro -v "$WORK/preflight.py":/preflight.py:ro "$IMG" /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE outer $(date +%T) =="
