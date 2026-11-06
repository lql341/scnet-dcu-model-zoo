#!/bin/bash -l
# 量化可行性探针：vLLM 在 DCU 上能不能跑 **AWQ（weight-only INT4）**？
# 背景：fp8（在线量化）在 DCU vLLM 上 engine core 直接起不来（gotchas #22）；
#       AWQ/GPTQ 走的是另一套 Triton 反量化 kernel，值得单独试一次。
# 模型：Qwen/Qwen3-8B-AWQ（6.1 GB，ModelScope 可得）
# 测：能否起服务 + 4K/c8 与 4K/c64 吞吐 + 抽 8 题看输出是否正常（不评分，只看是否退化到胡言乱语）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J awq-probe
#SBATCH -o /public/home/scnethr2601a10/work/awq-probe-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
CLIIMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
MODELHOST=/public/home/scnethr2601a10/models/Qwen3-8B-AWQ
WORK=/public/home/scnethr2601a10/work/awq-probe
PORT=30000
mkdir -p "$WORK/bench"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIB=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
$PD image exists "$CLIIMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$CLIIMG" >/dev/null 2>&1
echo "[host] run $(date +%T)"
$PD run -d --name srv --network=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/AWQ:ro -v "$WORK":/work -e LIB="$LIB" \
  "$IMG" -c 'export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; exec python3 -m vllm.entrypoints.openai.api_server --model /models/AWQ --served-model-name qwen3-8b-awq --host 127.0.0.1 --port 30000 --trust-remote-code --max-model-len 20480 --gpu-memory-utilization 0.95 --max-num-seqs 64' >/dev/null 2>&1
OK=0
for t in $(seq 1 150); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; echo "[host] 服务就绪 $((t*10))s"; break; }
  $PD ps --format '{{.Names}}' | grep -q '^srv$' || { echo "[host] 容器退出"; break; }
  sleep 10
done
$PD logs srv > "$WORK/server.log" 2>&1
if [ $OK -ne 1 ]; then echo "[host] 启动失败，关键行："; grep -iE "error|not supported|unsupported|quantiz|kernel" "$WORK/server.log" | tail -8 | cut -c1-200; $PD rm -f srv >/dev/null 2>&1; exit 1; fi
echo "[host] 量化配置："; grep -oiE "quantization=[a-z0-9_]+|awq|gptq|marlin" "$WORK/server.log" | sort -u | head -5 | sed 's/^/  /'
echo "[host] 生成抽查（8 条，检查是否退化）"
curl -sf --noproxy '*' "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-8b-awq","messages":[{"role":"user","content":"用一句话解释什么是张量并行。"}],"temperature":0,"max_tokens":120}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('  回答:',d['choices'][0]['message']['content'][:160].replace(chr(10),' '))" 2>&1 | tail -2
echo "[host] 压测（4K/c8、4K/c64）"
for C in 8 64; do
  N=$(( C*2 > 8 ? C*2 : 8 )); F="$WORK/bench/awq_4096_c$C.jsonl"; rm -f "$F"
  $PD run --rm --network=host --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$MODELHOST":/models/AWQ:ro "$CLIIMG" -c "export LD_LIBRARY_PATH=$LIB:\$LD_LIBRARY_PATH; cd /work && python3 -m sglang.bench_serving --backend vllm --host 127.0.0.1 --port 30000 --model qwen3-8b-awq --tokenizer /models/AWQ --dataset-name random-ids --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 --max-concurrency $C --num-prompts $N --seed 1 --output-file '$F'" 2>&1 | tail -2
  [ -s "$F" ] && python3 -c "
import json
r=json.loads([x for x in open('$F').read().split(chr(10)) if x.strip()][-1])
print('  c%s out=%.2f tok/s tpot=%.1f' % (r['max_concurrency'], r['output_throughput'], r['mean_tpot_ms']))" || echo "  c$C FAILED"
done
$PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1
echo "[host] DONE $(date +%T)"
