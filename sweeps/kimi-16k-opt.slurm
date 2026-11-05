#!/bin/bash -l
# Kimi-Linear-48B 16K/c64 回滚修复：开 CUDA Graph 后该档从 249.9 → 234.17（−6.3%）
# 假设：graph 默认按 max-running-requests 捕获过多 batch size，长上下文时挤占 KV cache
# 手段：--cuda-graph-max-bs 限制捕获上界 / 提高 mem-fraction-static / chunked-prefill
# 只测 16K/c64 与 16K/c8，1 轮；TP=4（权重 91.5 G，必须 4 卡）
# 注意：本集群 sbatch 命令行 --gres 覆盖无效，卡数只能改文件头
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -J kimi16k
#SBATCH -o /public/home/scnethr2601a10/work/kimi16k-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz
MODELHOST=/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct
WORK=/public/home/scnethr2601a10/work/kimi16k
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
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_ROCM_FUSED_DECODE_MLA=0
W=/work; M=/models/Kimi; NAME=kimi-linear-48b; PORT=30000
BENCH=sglang.benchmark.serving; python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>} ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>}"
run_variant() {
  local TAG=$1; shift
  echo ""; echo "########## $TAG  $*"
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 10
  python3 -m sglang.launch_server --model-path "$M" --served-model-name "$NAME" \
    --host 127.0.0.1 --port $PORT --tp 4 --attention-backend triton \
    --mem-fraction-static 0.85 --max-running-requests 64 --context-length 20480 "$@" > "$W/srv-$TAG.log" 2>&1 &
  local OK=0
  for t in $(seq 1 240); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; echo "  up $((t*5))s"; break; }
    sleep 5
  done
  [ $OK -ne 1 ] && { echo "  启动失败:"; grep -iE "error|not supported|Traceback|invalid device" "$W/srv-$TAG.log" | tail -3; return 1; }
  grep -oiE "cuda.graph[^,]*|max_bs=[0-9]+|mem_fraction_static=[0-9.]+" "$W/srv-$TAG.log" | sort -u | head -4 | sed 's/^/  cfg /'
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
  pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 15
}
run_variant "K0_graph_base"                      # 现状（graph 开，默认捕获）
run_variant "K1_graphmaxbs8"   --cuda-graph-max-bs 8
run_variant "K2_mem92"         --mem-fraction-static 0.92
run_variant "K3_chunk4k"       --chunked-prefill-size 4096
run_variant "K4_nograph"       --disable-cuda-graph   # 对照：历史最优（无 graph）
echo ""; echo "== DONE $(date +%T)"
EOS
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-}" \
  -v "$MODELHOST":/models/Kimi:ro -v "$WORK":/work -v "$WORK/inner.sh":/inner.sh:ro \
  "$IMG" /inner.sh 2>&1 | grep -viE "subuid|rootless|level=warning"
echo "[host] DONE $(date +%T)"
