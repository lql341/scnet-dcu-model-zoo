#!/bin/bash -l
# MiMo vLLM 配置扫描：验证 --enforce-eager 是否为瓶颈
# 每档只跑关键 3 个组合（4K/c1、4K/c64、16K/c8），1 轮，用于快速找出最优配置
# 找到最优后再跑一次完整 6 档 × 2 轮作为新的性能基准
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:30:00
#SBATCH -J hrvllmsweep
#SBATCH -o /public/home/scnethr2601a10/work/vsweep/vsweep-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vsweep/vsweep-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
TAR=""
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vsweep
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 32 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

[ -n "$TAR" ] && $PD load -i "$TAR" >/tmp/load.log 2>&1; $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
$PD image exists "$IMG" || { tail -5 /tmp/load.log; exit 1; }
echo "image ready $(date +%T)"

cat > /tmp/sweep_inner.sh <<'EOS'
set -uo pipefail
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving

run_variant() {
  local TAG="$1"; shift
  local EXTRA="$*"
  echo ""
  echo "########## $TAG"
  echo "   args: $EXTRA"
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 8
  python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$NAME" \
    --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
    --max-model-len 20480 $EXTRA \
    > "$W/server-$TAG.log" 2>&1 &
  local SPID=$! OK=0
  # 只以 /health 为准：vLLM 的 api_server 会 fork 出 APIServer/EngineCore 子进程，
  # 父进程 PID 提前退出属正常，不能用 kill -0 判定服务存活。
  for t in $(seq 1 120); do
    if curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
      OK=1; echo "  up after $((t*10))s"; break
    fi
    sleep 10
  done
  if [ $OK -ne 1 ]; then
    echo "  FAILED_TO_START; 关键报错:"
    grep -iE "error|out of memory|no available|not supported|Traceback" "$W/server-$TAG.log" | tail -4
    return 1
  fi
  mkdir -p "$W/bench"
  local ROWS=""
  for SPEC in "4096 1" "4096 64" "16384 8"; do
    set -- $SPEC; local ILEN=$1 C=$2
    local N=$(( C * 2 > 8 ? C * 2 : 8 ))
    local F="$W/bench/${TAG}_${ILEN}_c${C}.jsonl"
    rm -f "$F"
    timeout 1500 python3 -m "$BENCH" --backend vllm --host 127.0.0.1 --port $PORT \
      --model "$NAME" --tokenizer "$MODEL" --dataset-name random-ids \
      --random-input-len $ILEN --random-output-len 1024 --random-range-ratio 1 \
      --max-concurrency $C --num-prompts $N --seed 1 \
      --output-file "$F" > "$W/bench/${TAG}_${ILEN}_c${C}.log" 2>&1
    if [ -s "$F" ]; then
      R=$(python3 -c "
import json
r=json.loads(open('$F').read().strip().split(chr(10))[-1])
print('%s in / c%-3s  out=%8.2f tok/s  ttft=%9.1f ms  e2e=%10.1f ms  done=%s' % (r['random_input_len'], r['max_concurrency'], r['output_throughput'], r['mean_ttft_ms'], r['mean_e2e_latency_ms'], r['completed']))
")
      echo "  $R"; ROWS="$ROWS$R"$'\n'
    else
      echo "  ${ILEN}/c${C} FAILED"
    fi
  done
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 8
  echo "$ROWS" > "$W/result-$TAG.txt"
}

run_variant "W1_base_enforce"   "--gpu-memory-utilization 0.97 --enforce-eager"
run_variant "W2_cudagraph"      "--gpu-memory-utilization 0.97"
run_variant "W3_cg_seqs32"      "--gpu-memory-utilization 0.95 --max-num-seqs 32"

echo ""
echo "########## 汇总"
for f in "$W"/result-W*.txt; do
  [ -s "$f" ] || continue
  echo "--- $(basename "$f" .txt | sed 's/^result-//')"
  cat "$f"
done
EOS

echo "== run $(date +%T) =="
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  -v /tmp/sweep_inner.sh:/sweep_inner.sh:ro "$IMG" /sweep_inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
