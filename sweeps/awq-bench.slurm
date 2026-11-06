#!/bin/bash -l
# AWQ 在 DCU 上可用（2026-11-06 证实：awq_marlin + TritonW4A16LinearKernel 起服务、生成正常）
# 本脚本补压测：4K/c1、4K/c8、4K/c64 三档，与同规模 bf16 模型（MiMo-9B 是 9B，Qwen3-8B 是 8B）对比
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J awq-bench
#SBATCH -o /public/home/scnethr2601a10/work/awq-bench-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-qwen_asr1.7_bw:1.0.0
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
CLIIMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
MODELHOST=/public/home/scnethr2601a10/models/Qwen3-8B-AWQ
WORK=/public/home/scnethr2601a10/work/awq-bench
PORT=30000
mkdir -p "$WORK/bench"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIB=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
for I in "$IMG" "$CLIIMG"; do
  $PD image exists "$I" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$I" >/dev/null 2>&1
done
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
[ $OK -ne 1 ] && { echo "[host] 启动失败"; $PD logs srv 2>&1 | tail -5; exit 1; }
# 客户端脚本：写到 work 里再执行，避免外层引号把 $ 展开
cat > "$WORK/client.sh" <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
cd /work
for C in 1 8 64; do
  N=$(( C*2 > 8 ? C*2 : 8 ))
  python3 -m sglang.bench_serving --backend vllm --host 127.0.0.1 --port 30000 \
    --model qwen3-8b-awq --tokenizer /models/AWQ --dataset-name random-ids \
    --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
    --max-concurrency $C --num-prompts $N --seed 1 \
    --output-file "/work/bench/awq_4096_c$C.jsonl" > "/work/bench/awq_4096_c$C.log" 2>&1
  echo "  c$C rc=$?"
done
EOS
$PD run --rm --network=host --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$MODELHOST":/models/AWQ:ro "$CLIIMG" /work/client.sh 2>&1 | tail -5
echo "[host] 结果："
for f in "$WORK"/bench/*.jsonl; do [ -e "$f" ] && python3 -c "
import json
r=json.loads([x for x in open('$f').read().split(chr(10)) if x.strip()][-1])
print('  %s: out=%.2f tok/s ttft=%.0f ms tpot=%.1f ms done=%s' % ('$(basename $f .jsonl)', r['output_throughput'], r['mean_ttft_ms'], r['mean_tpot_ms'], r['completed']))"; done
echo "[host] 失败日志（若有）："; for f in "$WORK"/bench/*.log; do s=$(stat -c%s "$f"); [ "$s" -gt 0 ] && { echo "  == $(basename $f)"; tail -3 "$f" | cut -c1-160; }; done
$PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1
echo "[host] DONE $(date +%T)"
