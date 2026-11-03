#!/bin/bash -l
# MiMo vLLM 长上下文专项扫描 v3：注意力后端 / KV 精度 / TP=2
#
# v2 结论（2026-11-03，2 卡 1.5 h）：
#   X1 base / X2 batch8192 / X3 batch2048 / X4 VLLM_ROCM_USE_AITER=1
#   四者在 16K/c1、16K/c8、4K/c64 上**完全一致**（8.20 / 44.9 / 492）
#   → chunked-prefill 切片大小与 AITER 开关都不是瓶颈；TPOT 恒为 119 ms。
#   X5 TP=2 启动失败，根因：`local_world_size (2) must be <= visible devices (1)`
#   —— 容器没有继承 Slurm 的 HIP_VISIBLE_DEVICES（podman 默认不传环境变量）。
#   本版修掉该问题，并改测「注意力后端」「KV 缓存精度」两条真正能改 decode 的路径。
#
# v2 原头部注释保留于 git 历史。
#
# 为什么要 v2：vLLM 镜像里**没有 sglang**，压测客户端（sglang.benchmark.serving）
# 必须跑在 SGLang 镜像里；v1 把两者放同一个容器，三档全部 FAILED
# （`ModuleNotFoundError: No module named 'sglang'`）。
#
# 调查目标（来自 server.log 的实测线索）：
#   WARNING [chunked_prefill_paged_decode.py:400] Cannot use ROCm custom paged attention kernel,
#           falling back to Triton implementation.
#   结果：16K/c1 8.19 tok/s（SGLang 同模型 41.47）、16K/c8 44.83（SGLang 308.72）。
#   本扫描测：prefill 切片大小、AITER 内核、TP=2 能否把这个洞补上。
#
# 提交：sbatch sweep/sweep_16k2.slurm
# 2026-11-03 11:55 修正：`env $E_ENVS exec python3` 在 E_ENVS 为空时会把 `exec` 当命令名
#   （`env: 'exec': No such file or directory`，5 个变体全部秒退）→ 改为 shell 内 export + exec。
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:2
#SBATCH -c 16
#SBATCH -t 03:30:00
#SBATCH -J mimovllm16k3
#SBATCH -o /public/home/scnethr2601a10/work/vsweep16k2/sweep-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vsweep16k2/sweep-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
CLIIMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
CLITAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vsweep16k3
PORT=30000
mkdir -p "$WORK"/bench
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIB='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 70 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

$PD image exists "$IMG" 2>/dev/null || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
$PD image exists "$IMG" || { echo "server image unavailable"; tail -5 /tmp/load.log; exit 1; }
[ -s "$CLITAR" ] && $PD load -i "$CLITAR" >/tmp/load-cli.log 2>&1
$PD image exists "$CLIIMG" || { echo "cli image unavailable"; exit 1; }
echo "images ready $(date +%T)"

# ---------- 压测客户端脚本（SGLang 镜像内执行） ----------
cat > "$WORK/tiers.sh" <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
TAG=$1; PORT=$2; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; W=/work
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
for SPEC in "16384 8" "16384 1"; do
  set -- $SPEC; ILEN=$1; C=$2
  N=$(( C * 2 > 8 ? C * 2 : 8 ))
  F="$W/bench/${TAG}_${ILEN}_c${C}.jsonl"; rm -f "$F"
  timeout 1800 python3 -m "$BENCH" --backend vllm --host 127.0.0.1 --port "$PORT" \
    --model "$NAME" --tokenizer "$MODEL" --dataset-name random-ids \
    --random-input-len "$ILEN" --random-output-len 1024 --random-range-ratio 1 \
    --max-concurrency "$C" --num-prompts "$N" --seed 1 \
    --output-file "$F" > "$W/bench/${TAG}_${ILEN}_c${C}.log" 2>&1
  if [ -s "$F" ]; then
    python3 - "$F" >> "$W/result-$TAG.txt" <<'PY'
import json,sys
l=[x for x in open(sys.argv[1]).read().strip().split("\n") if x.strip()][-1]
r=json.loads(l)
ok = r["mean_ttft_ms"]>0 and r["completed"]>0 and r["output_throughput"]<20000
print("  in=%-5s c%-3s out=%8.2f tok/s ttft=%9.1f ms tpot=%7.2f ms done=%s %s" % (
  r["random_input_len"], r["max_concurrency"], r["output_throughput"],
  r["mean_ttft_ms"], r["mean_tpot_ms"], r["completed"], "" if ok else "INVALID"))
PY
  else
    echo "  ${ILEN}/c${C} FAILED" >> "$W/result-$TAG.txt"
  fi
done
EOS
chmod 644 "$WORK/tiers.sh"

wait_vram_free() {  # 等服务进程真正退出并释放显存，最多 180 s
  for t in $(seq 1 60); do
    $PD ps -a --format '{{.Names}}' 2>/dev/null | grep -q '^srv$' || { sleep 5; return 0; }
    sleep 3
  done
}

run_variant() {   # $1=TAG $2=额外环境变量串 $3...=vLLM 参数
  local TAG=$1; local ENVS=$2; shift 2
  echo ""; echo "########## $TAG  env=${ENVS:-none}  args=$*"
  : > "$WORK/result-$TAG.txt"
  $PD rm -f srv >/dev/null 2>&1
  $PD run -d --name srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
    -e E_LIB="$LIB" -e E_ENVS="$ENVS" \
    -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
    -e ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-}" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$E_LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; if [ -n "${E_ENVS:-}" ]; then export $E_ENVS; fi; exec python3 -m vllm.entrypoints.openai.api_server --model /models/MiMo --served-model-name mimo-v2.6-9b --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 --max-model-len 20480 '"$*" > /dev/null 2>&1
  local OK=0
  for t in $(seq 1 150); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "  up after $((t*5))s"; break; }
    $PD ps --format '{{.Names}}' | grep -q '^srv$' || { echo "  container exited"; break; }
    sleep 5
  done
  $PD logs srv > "$WORK/server-$TAG.log" 2>&1
  grep -oE "enforce_eager=[A-Za-z]+|max_num_batched_tokens=[0-9]+|cudagraph_mode=<[A-Za-z.]+: [0-9]>|tensor_parallel_size=[0-9]+" "$WORK/server-$TAG.log" | sort -u | sed 's/^/  cfg /'
  grep -i "paged attention" "$WORK/server-$TAG.log" | tail -1 | sed 's/^/  attn /'
  if [ $OK -ne 1 ]; then
    grep -iE "error|out of memory|less than desired|not supported|ValueError" "$WORK/server-$TAG.log" | tail -3 | sed 's/^/  ERR /'
    $PD rm -f srv >/dev/null 2>&1; wait_vram_free; return 1
  fi
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
    -v "$WORK":/work -v "$MODELHOST":/models/MiMo:ro \
    "$CLIIMG" /work/tiers.sh "$TAG" "$PORT" 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -5
  cat "$WORK/result-$TAG.txt"
  $PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1; wait_vram_free
}

BASE=(--gpu-memory-utilization 0.95 --max-num-seqs 128)
run_variant "Y1_flashattn"  "VLLM_ATTENTION_BACKEND=FLASH_ATTN"   "${BASE[@]}"
run_variant "Y2_kvfp8"      ""                                     "${BASE[@]}" --kv-cache-dtype fp8
run_variant "Y3_flash_kvfp8" "VLLM_ATTENTION_BACKEND=FLASH_ATTN"  "${BASE[@]}" --kv-cache-dtype fp8
run_variant "Y4_tp2"        ""                                     --gpu-memory-utilization 0.95 --max-num-seqs 128 --tensor-parallel-size 2

echo ""; echo "########## 汇总 $(date +%T)"
for f in "$WORK"/result-Y*.txt; do
  [ -s "$f" ] || continue
  echo "--- $(basename "$f" .txt | sed 's/^result-//')"; cat "$f"
done
echo "== DONE $(date +%T) =="
