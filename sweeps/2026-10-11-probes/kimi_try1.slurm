#!/bin/bash -l
# Kimi-Linear-48B-A3B-Instruct on BW1000 + SGLang 0.5.10（TP=4）
# 第 1 关：确认镜像自带 kimi_linear + fla/kda，能起服务
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 00:50:00
#SBATCH -J hrkimi
#SBATCH -o /public/home/scnethr2601a10/work/kimi/try1-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/kimi/try1-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct
WORK=/public/home/scnethr2601a10/work/kimi
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }

echo "== load image =="
S=$(date +%s)
$PD load -i "$TAR" >/tmp/load.log 2>&1 || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
echo "load_rc=$? $(( $(date +%s) - S ))s"
$PD image exists "$IMG" || { tail -5 /tmp/load.log; exit 1; }

cat > /tmp/inner.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_ROCM_FUSED_DECODE_MLA=0
W=/work; MODEL=/models/Kimi; NAME=kimi-linear-48b; PORT=30000

echo "=== 设备 ==="
python3 -c "
import torch
print('torch:', torch.__version__, 'hip:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i); print('  [%d] %s %.1fGB %s' % (i,p.name,p.total_memory/2**30,p.gcnArchName))
" 2>&1 | tail -8

echo "=== 框架是否自带 kimi_linear / fla ==="
python3 - <<'PY' 2>&1 | tail -12
mods = ["sglang.srt.models.kimi_linear", "sglang.srt.configs.kimi_linear",
        "sglang.srt.layers.attention.fla.kda", "sglang.srt.layers.attention.fla.chunk",
        "sglang.srt.layers.attention.fla.fused_norm_gate", "fla", "einops"]
for m in mods:
    try:
        __import__(m); print("  OK   ", m)
    except Exception as e:
        print("  FAIL ", m, type(e).__name__, str(e)[:80])
PY

python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { echo DEVICE_FAIL; exit 1; }

echo "=== launch sglang TP=4 $(date +%T) ==="
cd /root
python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$NAME" \
  --tp-size 4 --context-length 8192 --mem-fraction-static 0.85 \
  --max-running-requests 8 \
  --host 127.0.0.1 --port $PORT --trust-remote-code \
  --attention-backend triton --disable-cuda-graph \
  > "$W/server.log" 2>&1 &
SPID=$!
OK=0
for t in $(seq 1 150); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_EXITED at $((t*10))s"; break; }
  curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { OK=1; echo "SERVER_UP after $((t*10))s $(date +%T)"; break; }
  sleep 10
done
if [ $OK -ne 1 ]; then
  echo "--- 关键报错 ---"
  grep -inE "error|Exception|assert|out of memory|no available|not supported|Traceback" "$W/server.log" | tail -15
  echo "--- tail ---"; tail -25 "$W/server.log"
  exit 1
fi

echo "=== smoke ==="
{
  echo "== /health"; curl -sS -f --noproxy '*' -m 60 -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:$PORT/health
  echo "== /v1/models"; curl -sS -f --noproxy '*' -m 60 http://127.0.0.1:$PORT/v1/models; echo
  echo "== 单轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"kimi-linear-48b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":96}'; echo
  echo "== 多轮"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"kimi-linear-48b","messages":[{"role":"user","content":"我叫小海。"},{"role":"assistant","content":"你好，小海！"},{"role":"user","content":"我叫什么名字？"}],"temperature":0,"max_tokens":96}'; echo
  echo "== 流式"; curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"kimi-linear-48b","messages":[{"role":"user","content":"用一句话介绍国产异构加速器。"}],"temperature":0,"max_tokens":64,"stream":true}' | tail -3
} > "$W/smoke.log" 2>&1
echo "smoke_rc=$?"; tail -6 "$W/smoke.log"
hy-smi 2>/dev/null | grep -E "^[0-9] " | head -5
echo "--- server tail ---"; tail -8 "$W/server.log"
kill $SPID 2>/dev/null
echo "=== DONE inner $(date +%T) ==="
EOS

echo "== run $(date +%T) =="
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/Kimi:ro -v "$WORK":/work \
  -v /tmp/inner.sh:/inner.sh:ro "$IMG" /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE outer $(date +%T) =="
