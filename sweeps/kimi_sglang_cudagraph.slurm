#!/bin/bash -l
# Kimi-Linear SGLang 配置扫描：验证 CUDA graph 能否开启（TP=4）
# 每档只跑关键 3 个组合（4K/c1、4K/c64、16K/c8），1 轮，用于快速找出最优配置
# 找到最优后再跑一次完整 6 档 × 2 轮作为新的性能基准
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J hrkimisweep
#SBATCH -o /public/home/scnethr2601a10/work/ksweep/ksweep-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/ksweep/ksweep-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct
WORK=/public/home/scnethr2601a10/work/ksweep
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

$PD load -i "$TAR" >/tmp/load-$JID.log 2>&1 || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load-$JID.log 2>&1
$PD image exists "$IMG" || { tail -5 /tmp/load-$JID.log; exit 1; }
echo "image ready $(date +%T)"

cat > /tmp/sweep_inner.sh <<'EOS'
set -uo pipefail
# Slurm 的 HIP/CUDA_VISIBLE_DEVICES 是相对索引，与 rootless 容器内的
# renderD* 映射不一致，TP>1 时会导致 'HIP error: invalid device ordinal'。
# 清掉让 HIP 自行枚举容器内可见设备。
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_ROCM_FUSED_DECODE_MLA=0
W=/work; MODEL=/models/Kimi; NAME=kimi-linear-48b; PORT=30000
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving

run_variant() {
  local TAG="$1"; shift
  local EXTRA="$*"
  echo ""
  echo "########## $TAG"
  echo "   args: $EXTRA"
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 5
  python3 -m sglang.launch_server \
    --model-path "$MODEL" --served-model-name "$NAME" \
    --tp-size 4 --context-length 20480 --trust-remote-code \
    --host 127.0.0.1 --port $PORT $EXTRA \
    > "$W/server-$TAG.log" 2>&1 &
  local SPID=$! OK=0
  for t in $(seq 1 90); do
    kill -0 $SPID 2>/dev/null || { echo "  SERVER_EXITED at $((t*10))s"; break; }
    curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { OK=1; echo "  up after $((t*10))s"; break; }
    sleep 10
  done
  if [ $OK -ne 1 ]; then
    echo "  FAILED_TO_START; 关键报错:"; grep -iE "error|out of memory|no available|assert" "$W/server-$TAG.log" | tail -4
    return 1
  fi
  mkdir -p "$W/bench"
  local ROWS=""
  for SPEC in "4096 1" "4096 64" "16384 8"; do
    set -- $SPEC; local ILEN=$1 C=$2
    local N=$(( C * 2 > 8 ? C * 2 : 8 ))
    local F="$W/bench/${TAG}_${ILEN}_c${C}.jsonl"
    rm -f "$F"
    timeout 1500 python3 -m "$BENCH" --backend sglang --host 127.0.0.1 --port $PORT \
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
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 5
  echo "$ROWS" > "$W/result-$TAG.txt"
}

run_variant "K1_base_nocudagraph" "--mem-fraction-static 0.85 --max-running-requests 64 --attention-backend triton --disable-cuda-graph"
run_variant "K2_cudagraph"        "--mem-fraction-static 0.85 --max-running-requests 64 --attention-backend triton"

echo ""
echo "########## 汇总"
for f in "$W"/result-V*.txt; do
  [ -s "$f" ] || continue
  echo "--- $(basename "$f" .txt | sed 's/^result-//')"
  cat "$f"
done
EOS

echo "== run $(date +%T) =="
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/Kimi:ro -v "$WORK":/work \
  -v /tmp/sweep_inner.sh:/sweep_inner.sh:ro "$IMG" /sweep_inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
