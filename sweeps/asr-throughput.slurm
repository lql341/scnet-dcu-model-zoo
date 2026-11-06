#!/bin/bash -l
# ASR 离线批吞吐优化（T04 唯一的性能参考缺口）：同一 60 条子集，对比不同并发/批参数
#   A 现状：max-num-seqs 16 + 客户端 4 并发（官方 300 条就是这个配置，30 s）
#   B: max-num-seqs 64 + 客户端 16 并发
#   C: max-num-seqs 128 + 客户端 32 并发
# 指标：60 条总耗时、音频秒/墙钟秒（RTF）、CER（应不变）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -J asr-tput
#SBATCH -o /public/home/scnethr2601a10/work/asr-tput-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
MODELHOST=/public/home/scnethr2601a10/models/Qwen3-ASR-1.7B
WORK=/public/home/scnethr2601a10/work/asr-tput
COND=/public/home/scnethr2601a10/work/official-conds/qwen3_asr.jsonl
PORT=30000
mkdir -p "$WORK/repo"; export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
LIB=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }
head -20 "$COND" > "$WORK/repo/subset.jsonl"; sed -n '101,140p' "$COND" >> "$WORK/repo/subset.jsonl"
cp -f /public/home/scnethr2601a10/work/asr-quant-src/evaluate_official.py "$WORK/repo/" 2>/dev/null || true
[ -f "$WORK/repo/evaluate_official.py" ] || { echo "[host] 缺 evaluate_official.py"; exit 3; }

run_cfg() {  # $1=TAG $2=max_num_seqs $3=workers
  local TAG=$1 MNS=$2 W=$3
  echo ""; echo "########## $TAG  max_num_seqs=$MNS workers=$W"
  $PD rm -f srv >/dev/null 2>&1; sleep 3
  $PD run -d --name srv --network=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
    -v "$MODELHOST":/models/ASR:ro -v "$WORK":/work -e LIB="$LIB" -e MNS="$MNS" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; exec python3 -m vllm.entrypoints.openai.api_server --model /models/ASR --served-model-name qwen3-asr-1.7b --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 --max-num-seqs "$MNS" --gpu-memory-utilization 0.95' >/dev/null 2>&1
  local OK=0
  for t in $(seq 1 120); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { OK=1; echo "  服务就绪 $((t*10))s"; break; }
    $PD ps --format '{{.Names}}' | grep -q '^srv$' || { echo "  容器退出"; break; }
    sleep 10
  done
  [ $OK -ne 1 ] && { echo "  启动失败"; $PD logs srv 2>&1 | tail -3; $PD rm -f srv >/dev/null 2>&1; return 1; }
  local T0=$(date +%s)
  $PD run --rm --network=host --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$MODELHOST":/models/ASR:ro \
    -v /public/home/scnethr2601a10/work/official-t34:/assets:ro -e LIB="$LIB" -e W="$W" -e TAG="$TAG" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; cd /work/repo && COND_FILE=subset.jsonl OUTDIR=out-'"$TAG"' ASSET_ROOT=/assets PORT=30000 WORKERS="$W" python3 evaluate_official.py 2>&1 | tail -8'
  local T1=$(date +%s)
  echo "  用时 $((T1-T0)) s（60 条）"
  $PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1; sleep 10
}
run_cfg "A_mns16_w4"   16 4
run_cfg "B_mns64_w16"  64 16
run_cfg "C_mns128_w32" 128 32
echo ""; echo "== DONE $(date +%T)"
