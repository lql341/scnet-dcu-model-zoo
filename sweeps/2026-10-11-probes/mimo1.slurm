#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 00:35:00
#SBATCH -J hrmimo
#SBATCH -o /public/home/scnethr2601a10/probe/mimo-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/mimo-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
DTKHOST=/public/software/compiler/dtk-26.04
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }
ls "$MODELHOST" | head -5; du -sh "$MODELHOST"

echo "== load image =="
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? elapsed=$(( $(date +%s) - START ))s"

cat > /tmp/inner.sh <<'EOS'
set -u
export LD_LIBRARY_PATH=/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/NanoLog/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo "--- device check ---"
python3 -c "
import torch
print('torch:', torch.__version__, 'hip:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print('  [%d] %s vram=%.1fGB arch=%s cu=%s' % (i,p.name,p.total_memory/2**30,getattr(p,'gcnArchName','?'),getattr(p,'multi_processor_count','?')))
" 2>&1 | tail -8

if python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "DEVICE_OK=1"
else
  echo "DEVICE_OK=0"; exit 0
fi

echo "--- launch sglang ---"
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
cd /root
nohup python3 -m sglang.launch_server \
  --model-path /models/MiMo \
  --served-model-name mimo-v2.6-9b \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code \
  --attention-backend triton \
  --mem-fraction-static 0.8 \
  --max-running-requests 16 \
  --context-length 4096 \
  --disable-cuda-graph > /tmp/server.log 2>&1 &
SPID=$!
for t in $(seq 1 90); do
  if ! kill -0 $SPID 2>/dev/null; then echo "SERVER_EXITED at ${t}0s"; break; fi
  if curl -sf --noproxy '*' -m 5 http://127.0.0.1:30000/health >/dev/null 2>&1; then
    echo "SERVER_UP after $((t*10))s"
    echo "--- smoke ---"
    curl -s --noproxy '*' -m 300 -X POST http://127.0.0.1:30000/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d '{"model":"mimo-v2.6-9b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"max_tokens":96,"temperature":0}' | head -c 900
    echo
    curl -s --noproxy '*' -m 60 http://127.0.0.1:30000/v1/models | head -c 300
    echo
    break
  fi
  sleep 10
done
echo "--- server.log tail ---"
tail -25 /tmp/server.log
kill $SPID 2>/dev/null
EOS

echo "== run =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v "$DTKHOST":/opt/dtk:ro -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/MiMo:ro \
  -v /tmp/inner.sh:/inner.sh:ro "$IMG" bash /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
