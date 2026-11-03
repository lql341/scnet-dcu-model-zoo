#!/bin/bash -l
# MiMo vLLM 长上下文专项扫描：为什么 16K 档比 SGLang 差 5–7 倍？
#
# 已知证据（2026-11-03 复查 server.log）：
#   WARNING [chunked_prefill_paged_decode.py:400] Cannot use ROCm custom paged attention kernel,
#           falling back to Triton implementation.
#   → decode 注意力退回 Triton；16K/c1 TPOT 119 ms（4K/c1 只有 47.9 ms），16K/c1 吞吐 8.19 tok/s
#     vs 同模型 SGLang 41.47 tok/s。怀疑点：paged-attn 内核缺失 / prefill 未切片 / 单卡带宽。
#
# 本扫描只测「有区分度」的三档（16K/c1、16K/c8、4K/c64），1 轮；最优组合再跑全量 6 档。
# 提交：sbatch sweep/sweep_16k.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:2          # X5 要测 TP=2，所以申请 2 卡（其余变体只用 0 号卡）
#SBATCH -c 16
#SBATCH -t 03:30:00
#SBATCH -J mimovllm16k
#SBATCH -o /public/home/scnethr2601a10/work/vsweep16k/sweep-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vsweep16k/sweep-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
CLIIMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
CLITAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vsweep16k
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 70 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

# 服务镜像：本地无 tar，直接 pull（4–6 min）；压测客户端镜像走本地 tar
$PD image exists "$IMG" 2>/dev/null || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
$PD image exists "$IMG" || { echo "server image unavailable"; tail -5 /tmp/load.log; exit 1; }
[ -s "$CLITAR" ] && $PD load -i "$CLITAR" >/tmp/load-cli.log 2>&1
echo "images ready $(date +%T)"

cat > /tmp/sweep16k_inner.sh <<'EOS'
set -uo pipefail
# 【注意】不要 unset HIP_VISIBLE_DEVICES：Slurm 设 0,1；清掉后 HIP 会去枚举宿主全部 8 卡而容器只映射了 2 个 → 越界
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo "  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving

bench_tier() {   # $1=TAG $2=inlen $3=conc
  local TAG=$1 ILEN=$2 C=$3
  local N=$(( C * 2 > 8 ? C * 2 : 8 ))
  local F="$W/bench/${TAG}_${ILEN}_c${C}.jsonl"
  rm -f "$F"
  timeout 1800 python3 -m "$BENCH" --backend vllm --host 127.0.0.1 --port $PORT \
    --model "$NAME" --tokenizer "$MODEL" --dataset-name random-ids \
    --random-input-len $ILEN --random-output-len 1024 --random-range-ratio 1 \
    --max-concurrency $C --num-prompts $N --seed 1 \
    --output-file "$F" > "$W/bench/${TAG}_${ILEN}_c${C}.log" 2>&1
  if [ -s "$F" ]; then
    python3 -c "
import json
l=[x for x in open('$F').read().strip().split(chr(10)) if x.strip()][-1]
r=json.loads(l)
# 有效性：TTFT 为 0 或吞吐离谱都当失败（客户端会把 400 当成功）
ok = r['mean_ttft_ms']>0 and r['completed']>0
print('  %s in/c%-3s out=%8.2f tok/s ttft=%9.1f ms tpot=%7.2f ms done=%s %s' % (
  r['random_input_len'], r['max_concurrency'], r['output_throughput'],
  r['mean_ttft_ms'], r['mean_tpot_ms'], r['completed'], '' if ok else 'INVALID'))
" | tee -a "$W/result-$TAG.txt"
  else
    echo "  ${ILEN}/c${C} FAILED" | tee -a "$W/result-$TAG.txt"
  fi
}

run_variant() {  # $1=TAG $2=env 赋值串（可空） $3...=vLLM 参数
  local TAG=$1; local ENVS=$2; shift 2
  echo ""; echo "########## $TAG"
  echo "   env: ${ENVS:-<none>}  args: $*"
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 10
  env $ENVS python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$NAME" \
    --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
    --max-model-len 20480 "$@" > "$W/server-$TAG.log" 2>&1 &
  local OK=0
  for t in $(seq 1 150); do
    if curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1; then
      OK=1; echo "  up after $((t*5))s"; break; fi
    sleep 5
  done
  if [ $OK -ne 1 ]; then
    echo "  FAILED_TO_START; 关键报错:"
    grep -iE "error|out of memory|no available|not supported|ValueError|Traceback" "$W/server-$TAG.log" | tail -4
    return 1
  fi
  # 记录本变体真实生效的引擎配置（事后核对用）
  grep -oE "enforce_eager=[A-Za-z]+|enable_prefix_caching=[A-Za-z]+|max_num_batched_tokens=[0-9]+|cudagraph_mode=<[A-Za-z.]+: [0-9]>" \
    "$W/server-$TAG.log" | sort -u | sed 's/^/  cfg /'
  grep -i "paged attention" "$W/server-$TAG.log" | tail -2 | sed 's/^/  attn /'
  : > "$W/result-$TAG.txt"
  bench_tier "$TAG" 16384 1
  bench_tier "$TAG" 16384 8
  bench_tier "$TAG" 4096 64
  pkill -9 -f "vllm.entrypoints" 2>/dev/null; sleep 10
}

mkdir -p "$W/bench"
BASE=(--gpu-memory-utilization 0.95 --max-num-seqs 128)

run_variant "X1_base"        ""                 "${BASE[@]}"
run_variant "X2_batch8192"   ""                 "${BASE[@]}" --max-num-batched-tokens 8192
run_variant "X3_batch2048"   ""                 "${BASE[@]}" --max-num-batched-tokens 2048
run_variant "X4_aiter"       "VLLM_ROCM_USE_AITER=1" "${BASE[@]}"
run_variant "X5_tp2"         ""                 --gpu-memory-utilization 0.95 --max-num-seqs 128 --tensor-parallel-size 2

echo ""; echo "########## 汇总"
for f in "$W"/result-X*.txt; do
  [ -s "$f" ] || continue
  echo "--- $(basename "$f" .txt | sed 's/^result-//')"; cat "$f"
done
EOS

echo "== run $(date +%T) =="
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  -v /tmp/sweep16k_inner.sh:/sweep16k_inner.sh:ro "$IMG" /sweep16k_inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

# 压测客户端（SGLang 镜像，vLLM 镜像没有 sglang.benchmark.serving）
# 说明：inner 脚本里已直接跑了 benchmark（该镜像自带 sglang），若上方日志显示缺 sglang 再用下面这段
echo "== DONE $(date +%T) =="
ls -la "$WORK"/result-X*.txt 2>/dev/null
