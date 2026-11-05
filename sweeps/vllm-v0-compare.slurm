#!/bin/bash -l
# vLLM v0 引擎对比：VLLM_USE_V1=0 能否绕开 16K 档的 Triton paged decode（gotchas #21 的下一步）
# 只测 16K/c8 + 16K/c1 两档，1 卡
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J vllm-v0
#SBATCH -o /public/home/scnethr2601a10/work/vllm-v0-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
CLIIMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
CLITAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vllm-v0
PORT=30000
mkdir -p "$WORK/bench"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIB='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }
# 本机未压缩 tar 已清理（25 GB 冗余），缺 tar 时直接 pull
if [ -s "$CLITAR" ]; then $PD load -i "$CLITAR" >/dev/null 2>&1; fi
$PD image exists "$CLIIMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$CLIIMG" >/dev/null 2>&1
$PD image exists "$CLIIMG" || { echo "[host] CLI image unavailable"; exit 1; }
cat > "$WORK/tiers.sh" <<'EOS'
set -uo pipefail
TAG=$1; PORT=$2; W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
BENCH=sglang.benchmark.serving; python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
for SPEC in "16384 8" "16384 1"; do
  set -- $SPEC; ILEN=$1; C=$2; N=$(( C*2 > 8 ? C*2 : 8 ))
  F="$W/bench/${TAG}_${ILEN}_c${C}.jsonl"; rm -f "$F"
  timeout 1800 python3 -m "$BENCH" --backend vllm --host 127.0.0.1 --port "$PORT" --model "$NAME" \
    --tokenizer "$MODEL" --dataset-name random-ids --random-input-len "$ILEN" --random-output-len 1024 \
    --random-range-ratio 1 --max-concurrency "$C" --num-prompts "$N" --seed 1 --output-file "$F" \
    > "$W/bench/${TAG}_${ILEN}_c${C}.log" 2>&1
  [ -s "$F" ] && python3 -c "
import json
r=json.loads([x for x in open('$F').read().split(chr(10)) if x.strip()][-1])
print('  in=%s c%-3s out=%8.2f tok/s tpot=%7.2f ms done=%s' % (r['random_input_len'], r['max_concurrency'], r['output_throughput'], r['mean_tpot_ms'], r['completed']))" || echo "  ${ILEN}/c${C} FAILED"
done
EOS
run_variant() {
  local TAG=$1; local ENVS=$2
  echo ""; echo "########## $TAG  env=$ENVS"
  $PD rm -f srv >/dev/null 2>&1; sleep 3
  $PD run -d --name srv --network=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
    -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work -e E_LIB="$LIB" -e E_ENVS="$ENVS" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$E_LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; if [ -n "${E_ENVS:-}" ]; then export $E_ENVS; fi; exec python3 -m vllm.entrypoints.openai.api_server --model /models/MiMo --served-model-name mimo-v2.6-9b --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 --max-model-len 20480 --gpu-memory-utilization 0.95 --max-num-seqs 128' >/dev/null 2>&1
  local OK=0
  for t in $(seq 1 120); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; echo "  up after $((t*5))s"; break; }
    $PD ps --format '{{.Names}}' | grep -q '^srv$' || { echo "  容器退出"; break; }
    sleep 5
  done
  $PD logs srv > "$WORK/server-$TAG.log" 2>&1
  if [ $OK -ne 1 ]; then echo "  启动失败:"; grep -iE "error|not supported|Engine core" "$WORK/server-$TAG.log" | tail -4; $PD rm -f srv >/dev/null 2>&1; return 1; fi
  grep -oE "disable_custom_all_reduce|V1|Running vLLM|vLLM version" "$WORK/server-$TAG.log" | sort -u | head -3 | sed 's/^/  cfg /'
  $PD run --rm --network=host --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$MODELHOST":/models/MiMo:ro \
    "$CLIIMG" /work/tiers.sh "$TAG" "$PORT" 2>&1 | grep -viE "subuid|rootless|level=warning" | tail -4
  $PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1; sleep 15
}
run_variant "V1_default" ""
run_variant "V0_engine"  "VLLM_USE_V1=0"
echo ""; echo "== DONE $(date +%T)"; ls -la "$WORK"/bench/*.jsonl 2>/dev/null | tail -4
