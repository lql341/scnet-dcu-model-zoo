#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 00:40:00
#SBATCH -J hrmimo3
#SBATCH -o /public/home/scnethr2601a10/probe/mimo3-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/mimo3-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
DTKHOST=/public/software/compiler/dtk-26.04
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }
echo "== load =="
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? $(( $(date +%s) - START ))s"

cat > /tmp/inner.sh <<'EOS'
set -u
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo "--- device ---"
python3 -c "
import torch
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print('dev0: %s vram=%.1fGB arch=%s' % (p.name,p.total_memory/2**30,getattr(p,'gcnArchName','?')))
    import time; x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(30): y=x@x
    torch.cuda.synchronize(); dt=time.time()-t
    print('bf16 4096^3: %.1f ms -> %.1f TFLOPS' % (dt/30*1000, 2*4096**3*30/dt/1e12))
" 2>&1 | tail -6
python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { echo DEVICE_FAIL; exit 0; }

echo "--- launch sglang $(date +%T) ---"
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_USE_AITER_FP8_ASM_MOE=1 \
       SGLANG_USE_TRITON_EXTEND_FROM_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
cd /root
nohup python3 -m sglang.launch_server \
  --model-path /models/MiMo --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 --trust-remote-code \
  --attention-backend triton --mem-fraction-static 0.8 \
  --max-running-requests 16 --context-length 4096 --disable-cuda-graph \
  > /tmp/server.log 2>&1 &
SPID=$!
OK=0
for t in $(seq 1 100); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_EXITED at $((t*10))s"; break; }
  if curl -sf --noproxy '*' -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "SERVER_UP after $((t*10))s  $(date +%T)"; OK=1; break
  fi
  sleep 10
done
if [ $OK -eq 1 ]; then
  echo "--- smoke 1: chat ---"
  curl -s --noproxy '*' -m 300 -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"max_tokens":96,"temperature":0}' | head -c 700
  echo; echo "--- smoke 2: gsm8k ---"
  curl -s --noproxy '*' -m 300 -X POST http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes. How much did she earn? Put the final answer in \\boxed{}."}],"max_tokens":600,"temperature":0}' | python3 -c "
import json,sys
d=json.load(sys.stdin)
c=d['choices'][0]['message']['content'] or ''
print('TAIL:', c[-200:].replace(chr(10),' '))
print('finish:', d['choices'][0]['finish_reason'], 'usage:', d['usage'])
"
  echo "--- vram ---"; /opt/hyhal/bin/hy-smi 2>/dev/null | grep -E "^[0-9]|HCU" | head -3
fi
echo "--- server.log tail ---"; tail -25 /tmp/server.log
kill $SPID 2>/dev/null
EOS

echo "== run =="
$PD run --rm --ipc=host --device /dev/kfd --device /dev/mkfd --device /dev/dri --privileged \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v "$DTKHOST":/opt/dtk:ro \
  -v "$MODELHOST":/models/MiMo:ro \
  -v /tmp/inner.sh:/inner.sh:ro "$IMG" bash /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
