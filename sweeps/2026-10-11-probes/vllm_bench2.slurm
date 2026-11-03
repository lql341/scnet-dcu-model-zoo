#!/bin/bash -l
# BW1000 × vLLM × MiMo 性能基准
#   容器 A（vLLM 镜像）: 起 vllm serve，--network=host 暴露到节点 127.0.0.1:30000
#   容器 B（SGLang 镜像）: 用官方 sglang.benchmark.serving 客户端打 localhost（vLLM 服务通用）
# 这样保持官方压测工具不变，只是把客户端放进另一个容器。
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J hrvbench3
#SBATCH -o /public/home/scnethr2601a10/work/vllm/bench2-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vllm/bench2-%j.err

set -uo pipefail
IMG_VLLM=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
IMG_SGL=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vllm
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 68 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }

for I in "$IMG_VLLM" "$IMG_SGL"; do
  echo "== pull $I $(date +%T) =="
  S=$(date +%s); $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$I" >/tmp/pull.log 2>&1
  echo "pull_rc=$? $(( $(date +%s) - S ))s"
  $PD image exists "$I" || { grep -iE "^Error" /tmp/pull.log | tail -3; exit 1; }
done
$PD images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | head -3

cleanup() { $PD rm -f vllm-srv >/dev/null 2>&1; }
trap cleanup EXIT

echo "== start vLLM 容器 $(date +%T) =="
$PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  "$IMG_VLLM" -c '
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /root
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/MiMo --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 \
  --max-model-len 20480 --gpu-memory-utilization 0.97 --enforce-eager
' > /dev/null 2>&1

OK=0
for t in $(seq 1 180); do
  curl -sf --noproxy '*' -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1 && { OK=1; echo "SERVER_UP after $((t*10))s $(date +%T)"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || { echo "CONTAINER_GONE"; break; }
  sleep 10
done
if [ $OK -ne 1 ]; then echo "SERVER_NOT_UP"; $PD logs vllm-srv 2>&1 | tail -15; exit 1; fi

echo "== preflight $(date +%T) =="
PORT=30000 NAME=mimo-v2.6-9b python3 "$WORK/preflight.py" || { echo PREFLIGHT_FAIL; exit 1; }

echo "== bench 客户端容器 $(date +%T) =="
cat > /tmp/client.sh <<'EOS'
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:$LD_LIBRARY_PATH
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000
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
python3 - <<'PY' > "$W/bench/summary.md" 2>&1
import json
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
EOS
$PD run --rm --network=host --entrypoint=/bin/bash \
  -v "$WORK":/work -v /tmp/client.sh:/client.sh:ro \
  -v "$MODELHOST":/models/MiMo:ro "$IMG_SGL" /client.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
