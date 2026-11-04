#!/bin/bash -l
# ASR 量化对照实验：bf16（基线）vs fp8（vLLM 在线量化），同一子集同一评估脚本
#   sbatch sweeps/asr-quant.slurm
# 子集：AISHELL 前 20 + LibriSpeech 前 40（共 60 条），跑两种精度各一次，比较 CER/WER 与耗时
# 说明：官方不要求评分；这里自算 CER/WER **只用于对比量化损失**，不是验收指标。
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J asr-quant
#SBATCH -o /public/home/scnethr2601a10/work/asr-quant-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/asr-quant-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
MODELHOST=/public/home/scnethr2601a10/models/Qwen3-ASR-1.7B
WORK=/public/home/scnethr2601a10/work/asr-quant
PORT=30000
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; exit 1; }

# 子集：AISHELL 前 20 + LibriSpeech 前 40（保持官方清单顺序，不改内容）
COND=/public/home/scnethr2601a10/work/official-conds/qwen3_asr.jsonl
[ -s "$COND" ] || { echo "[host] 缺 $COND"; exit 3; }
mkdir -p "$WORK/repo"
head -20 "$COND" > "$WORK/repo/subset.jsonl"
sed -n '101,140p' "$COND" >> "$WORK/repo/subset.jsonl"
wc -l "$WORK/repo/subset.jsonl"
cp -f /public/home/scnethr2601a10/work/asr-quant-src/evaluate_official.py "$WORK/repo/" 2>/dev/null
[ -f "$WORK/repo/evaluate_official.py" ] || cp -f /public/home/scnethr2601a10/work/zoo-asr-official/evaluate_official.py "$WORK/repo/" 2>/dev/null
echo "[host] run $(date +%T)"

run_prec() {   # $1=TAG $2=额外 vLLM 参数
  local TAG=$1; shift
  echo ""; echo "########## $TAG  额外参数: $*"
  $PD rm -f srv >/dev/null 2>&1; sleep 3
  $PD run -d --name srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/ASR:ro -v "$WORK":/work \
    -e LIB="$(echo /opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib)" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; exec python3 -m vllm.entrypoints.openai.api_server --model /models/ASR --served-model-name qwen3-asr-1.7b --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 --max-num-seqs 16 --gpu-memory-utilization 0.95 '"$*" > /dev/null 2>&1
  local OK=0
  for t in $(seq 1 180); do
    curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "  服务就绪 $((t*10))s"; break; }
    $PD ps --format '{{.Names}}' | grep -q '^srv$' || { echo "  容器退出"; break; }
    sleep 10
  done
  if [ $OK -ne 1 ]; then
    echo "  启动失败；server 日志末尾:"; $PD logs srv 2>&1 | tail -20 | cut -c1-200
    $PD rm -f srv >/dev/null 2>&1; return 1
  fi
  local T0=$(date +%s)
  $PD run --rm --network=host --entrypoint=/bin/bash --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$MODELHOST":/models/ASR:ro -v /public/home/scnethr2601a10/work/official-t34:/assets:ro \
    -e LIB="$(echo /opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib)" \
    "$IMG" -c 'export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"; export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal; cd /work/repo && COND_FILE=subset.jsonl OUTDIR=out-'"$TAG"' ASSET_ROOT=/assets PORT=30000 WORKERS=4 python3 evaluate_official.py 2>&1 | tail -6' 2>&1 | tail -12
  local T1=$(date +%s)
  echo "  评测耗时 $((T1-T0))s"
  $PD stop -t 10 srv >/dev/null 2>&1; $PD rm -f srv >/dev/null 2>&1; sleep 10
}

run_prec "bf16" ""
run_prec "fp8" "--quantization fp8"

echo ""; echo "########## 汇总"
for f in "$WORK"/repo/out-*/eval_result.json; do
  [ -s "$f" ] && { echo "--- $f"; python3 -c "import json,sys;d=json.load(open('$f'));print(json.dumps(d,ensure_ascii=False)[:400])" 2>/dev/null; }
done
echo "[host] DONE $(date +%T)"
