#!/bin/bash -l
# BW1000 × vLLM：拉镜像 → 起 vllm serve (MiMo) → smoke
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 00:45:00
#SBATCH -J hrvllm2
#SBATCH -o /public/home/scnethr2601a10/work/vllm/sbatch-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/vllm/sbatch-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work/vllm
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 60 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }

echo "== pull $(date +%T) =="
S=$(date +%s)
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" > /tmp/pull.log 2>&1
echo "pull_rc=$? $(( $(date +%s) - S ))s"
grep -iE "^Error|denied|not found" /tmp/pull.log | tail -3
$PD image exists "$IMG" || exit 1
$PD images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -2

cat > /tmp/inner.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
W=/work
echo "=== env.log ==="
{
  echo "image: image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
  hy-smi 2>/dev/null | head -12
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0); print('Device:', p.name, 'vram_GB:', round(p.total_memory/2**30,1), 'arch:', p.gcnArchName)
"
  python3 -c "import vllm,transformers,triton;print('vllm:',vllm.__version__);print('transformers:',transformers.__version__);print('triton:',triton.__version__)" 2>&1 | tail -4
} > "$W/env.log" 2>&1
cat "$W/env.log"
python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { echo DEVICE_FAIL; exit 1; }

echo "=== start vllm $(date +%T) ==="
cd /root
python3 -m vllm.entrypoints.openai.api_server \
  --model /models/MiMo --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 --trust-remote-code --dtype bfloat16 \
  --max-model-len 16384 --gpu-memory-utilization 0.95 --enforce-eager \
  > "$W/server.log" 2>&1 &
SPID=$!
OK=0
for t in $(seq 1 150); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_EXITED at $((t*10))s"; break; }
  curl -sf --noproxy '*' -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1 && { OK=1; echo "SERVER_UP after $((t*10))s $(date +%T)"; break; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "--- server.log ---"; grep -iE "error|Error|Traceback|No available memory|not supported" "$W/server.log" | tail -12; tail -15 "$W/server.log"; exit 1; }

echo "=== smoke ==="
{
  echo "== /health"; curl -sS -f --noproxy '*' -m 60 -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:30000/health
  echo "== /v1/models"; curl -sS -f --noproxy '*' -m 60 http://127.0.0.1:30000/v1/models; echo
  echo "== 单轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":96}'; echo
  echo "== 多轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"我叫小海。"},{"role":"assistant","content":"你好，小海！"},{"role":"user","content":"我叫什么名字？"}],"temperature":0,"max_tokens":96}'; echo
  echo "== 流式"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":64,"stream":true}' | tail -3
} > "$W/smoke.log" 2>&1
echo "smoke_rc=$?"
tail -6 "$W/smoke.log"
hy-smi 2>/dev/null | grep -E "^[0-9] " | head -2
echo "--- server.log tail ---"; tail -6 "$W/server.log"
kill $SPID 2>/dev/null
echo "=== DONE inner $(date +%T) ==="
EOS

echo "== run $(date +%T) =="
# 该镜像是 SothisAI Notebook 镜像，ENTRYPOINT=/bin/sh -c "<init>; sshd -D"，
# 必须用 --entrypoint 覆盖，否则命令被吞掉或报 "cannot execute binary file"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  -v /tmp/inner.sh:/inner.sh:ro "$IMG" /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE outer $(date +%T) =="
