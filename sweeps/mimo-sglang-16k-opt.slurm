#!/bin/bash -l
# MiMo × SGLang 16K 档优化：chunked-prefill 与 CUDA Graph 的**组合**从未一起测过
#   （历史扫描 V5 测了 chunked 但关了 graph；V4 开了 graph 但没 chunked）
# 只测 16K/c64 与 16K/c8 两档，1 轮
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:30:00
#SBATCH -J mimo16kopt
#SBATCH -o /public/home/scnethr2601a10/work/mimo16kopt-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/mimo16kopt
mkdir -p "$WORK/bench"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 40 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || { [ -s "$TAR" ] && gunzip -c "$TAR" | $PD load >/dev/null 2>&1; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }
cat > "$WORK/inner.sh" <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0
W=/work; M=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000
BENCH=sglang.benchmark.serving; python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
run_variant() {
  local TAG=$1; shift
  echo ""; echo "########## $TAG  $*"
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 8
  python3 -m sglang.launch_server --model-path "$M" --served-model-name "$NAME" \
    --host 127.0.0.1 --port $PORT --tp 1 --attention-backend triton \
    --mem-fraction-static 0.85 --max-running-requests 64 --context-length 20480 "$@" > "$W/srv-$TAG.log" 2>&1 &
  local OK=0
  for t in $(seq 1 180); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; echo "  up $((t*5))s"; break; }
    sleep 5
  done
  [ $OK -ne 1 ] && { echo "  启动失败:"; grep -iE "error|not supported|Traceback" "$W/srv-$TAG.log" | tail -3; return 1; }
  grep -oE "chunked_prefill|ChunkedPrefill|cuda_graph|CudaGraph|cuda graph" "$W/srv-$TAG.log" | sort -u | head -3 | sed 's/^/  cfg /'
  # 用 ${VAR% *}/${VAR#* } 拆参，避免 set -- 在函数里配合 set -u 时出现 "C: unbound variable"（踩过）
  for SPEC in "16384 64" "16384 8"; do
    local IL="${SPEC% *}"; local C="${SPEC#* }"
    local N=$(( C*2 > 8 ? C*2 : 8 ))
    local F="$W/bench/${TAG}_${IL}_c${C}.jsonl"; rm -f "$F"
    timeout 2400 python3 -m "$BENCH" --backend sglang --host 127.0.0.1 --port $PORT --model "$NAME" \
      --tokenizer "$M" --dataset-name random-ids --random-input-len $IL --random-output-len 1024 \
      --random-range-ratio 1 --max-concurrency $C --num-prompts $N --seed 1 --output-file "$F" \
      > "$W/bench/${TAG}_${IL}_c${C}.log" 2>&1
    [ -s "$F" ] && python3 -c "
import json
r=json.loads([x for x in open('$F').read().split(chr(10)) if x.strip()][-1])
ok = r['mean_ttft_ms']>0 and r['output_throughput']<20000
print('  in=%s c%-3s out=%8.2f tok/s ttft=%9.1f tpot=%7.2f %s' % (r['random_input_len'], r['max_concurrency'], r['output_throughput'], r['mean_ttft_ms'], r['mean_tpot_ms'], '' if ok else 'INVALID'))" || echo "  ${IL}/c${C} FAILED"
  done
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 10
}
run_variant "A_base_graph"        # 现行最优：graph 开、无 chunked
run_variant "B_chunk4k_graph"    --chunked-prefill-size 4096
run_variant "C_chunk8k_graph"    --chunked-prefill-size 8192
run_variant "D_mrr128_graph"     --max-running-requests 128
echo ""; echo "== DONE $(date +%T)"
EOS
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work -v "$WORK/inner.sh":/inner.sh:ro \
  "$IMG" /inner.sh 2>&1 | grep -viE "subuid|rootless|level=warning"
echo "[host] DONE $(date +%T)"
