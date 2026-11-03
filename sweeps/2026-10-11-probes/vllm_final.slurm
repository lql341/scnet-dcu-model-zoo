#!/bin/bash -l
# MiMo vLLM 全量基准（最优配置 W3：CUDA Graph 开启 + max-num-seqs 128）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:30:00
#SBATCH -J hrvllmopt
#SBATCH -o /public/home/scnethr2601a10/work/vopt/vopt-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vopt/vopt-%j.err

set -uo pipefail
IMG_VLLM=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
IMG_SGL=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vopt
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIBP='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 68 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

for I in "$IMG_VLLM" "$IMG_SGL"; do
  echo "== pull $I $(date +%T) =="
  $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$I" >/tmp/pull.log 2>&1
  $PD image exists "$I" || { echo "PULL_FAIL"; tail -3 /tmp/pull.log; exit 1; }
done
trap '$PD rm -f vsrv >/dev/null 2>&1' EXIT

cat > /tmp/cli.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=LIBP_PLACEHOLDER:$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
echo "bench_module=$BENCH"
mkdir -p "$W/bench"; rm -f "$W"/bench/round*.jsonl

echo "=== env.log ==="
{
  echo "image: image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
p=torch.cuda.get_device_properties(0); print('Device:', p.name, 'vram_GB:', round(p.total_memory/2**30,1), 'arch:', p.gcnArchName)
"
  python3 -c "import vllm,transformers,triton;print('vllm:',vllm.__version__);print('transformers:',transformers.__version__);print('triton:',triton.__version__)" 2>&1|tail -3
} > "$W/env.log" 2>&1
cat "$W/env.log"

echo "=== smoke ==="
{
  echo "== /health"; curl -sS -f --noproxy '*' -m 60 -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:$PORT/health
  echo "== /v1/models"; curl -sS -f --noproxy '*' -m 60 http://127.0.0.1:$PORT/v1/models; echo
  echo "== 单轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'$NAME'","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":256}'; echo
  echo "== 多轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'$NAME'","messages":[{"role":"user","content":"我叫小海。"},{"role":"assistant","content":"你好，小海！"},{"role":"user","content":"我叫什么名字？"}],"temperature":0,"max_tokens":256}'; echo
  echo "== 流式"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"'$NAME'","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":64,"stream":true}' | tail -3
} > "$W/smoke.log" 2>&1
echo "smoke_rc=$?"; tail -3 "$W/smoke.log"

echo "=== preflight ==="
PORT=$PORT NAME=$NAME python3 /preflight_lib.py || { echo PREFLIGHT_FAIL; exit 1; }

echo "=== bench 6档×2 ==="
for ILEN in 4096 16384; do
  for C in 1 8 64; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG $(date +%T)"
      timeout 2700 python3 -m "$BENCH" --backend vllm --host 127.0.0.1 --port $PORT \
        --model "$NAME" --tokenizer "$MODEL" --dataset-name random-ids \
        --random-input-len $ILEN --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency $C --num-prompts $N --seed 1 \
        --output-file "$W/bench/round${R}.jsonl" > "$W/bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG"
    done
  done
done
python3 /summarize.py
EOS
sed -i "s|LIBP_PLACEHOLDER|$LIBP|" /tmp/cli.sh

run_variant() {
  local TAG="$1"; shift
  echo ""; echo "########## $TAG"; echo "   args: $*"
  $PD rm -f vsrv >/dev/null 2>&1; sleep 3
  $PD run -d --name vsrv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro "$IMG_VLLM" -c "
export LD_LIBRARY_PATH=$LIBP:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /root
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/MiMo --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 \
  --max-model-len 20480 $* " > /dev/null 2>&1
  local OK=0
  for t in $(seq 1 150); do
    curl -sf --noproxy '*' -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1 && { OK=1; echo "  up after $((t*10))s"; break; }
    $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vsrv || { echo "  CONTAINER_GONE"; break; }
    sleep 10
  done
  if [ $OK -ne 1 ]; then
    echo "  FAILED_TO_START"; $PD logs vsrv 2>&1 | grep -iE "error|no available|not supported" | tail -4
    $PD rm -f vsrv >/dev/null 2>&1; return 1
  fi
  mkdir -p "$WORK/bench"
  $PD run --rm --network=host --entrypoint=/bin/bash \
    -v "$WORK":/work -v "$MODELHOST":/models/MiMo:ro -v /tmp/cli.sh:/cli.sh:ro -v "$WORK/preflight_lib.py":/preflight_lib.py:ro -v "$WORK/summarize.py":/summarize.py:ro \
    "$IMG_SGL" /cli.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
  $PD rm -f vsrv >/dev/null 2>&1
}

run_variant "FINAL" "--gpu-memory-utilization 0.95 --max-num-seqs 128"

echo ""; echo "########## 汇总"
for f in "$WORK"/result-W*.txt; do [ -s "$f" ] || continue; echo "--- $(basename "$f" .txt|sed 's/^result-//')"; cat "$f"; done
echo "== DONE $(date +%T) =="
