#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:30:00
#SBATCH -J hrbench
#SBATCH -o /public/home/scnethr2601a10/work/bench/sbatch-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/bench/sbatch-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
MODELHOST=/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B
WORK=/public/home/scnethr2601a10/work
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK/bench"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? $(( $(date +%s) - START ))s"

cat > /tmp/inner.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
export SGLANG_USE_LIGHTOP=1 SGLANG_KV_LAYOUT_DCU_FA=0 SGLANG_USE_AITER_FP8_ASM_MOE=1 \
       SGLANG_USE_TRITON_EXTEND_FROM_AITER=1 SGLANG_ROCM_FUSED_DECODE_MLA=0
W=/work; MODEL=/models/MiMo; NAME=mimo-v2.6-9b; PORT=30000; CTX=17408

python3 -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)" || { echo DEVICE_FAIL; exit 1; }

echo "=== env.log ==="
{
  echo "image: image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
  echo "dtk: $(cat /opt/dtk/.info/version 2>/dev/null || echo 26.04)  DTK_VERSION=$DTK_VERSION"
  /opt/hyhal/bin/hy-smi 2>/dev/null | head -12
  python3 -c "
import sys, torch
print('python:', sys.version.split()[0])
print('torch:', torch.__version__)
print('HIP:', torch.version.hip)
print('Device:', torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print('vram_GB:', round(p.total_memory/2**30, 1), 'arch:', getattr(p,'gcnArchName','?'))
"
  for m in sglang triton transformers; do python3 -c "import $m;print('$m:', getattr($m,'__version__','?'))" 2>/dev/null; done
  python3 -m pip list 2>/dev/null | grep -iE "^(torch|triton|sglang|transformers|tokenizers|flash)" | head -12
} > "$W/env.log" 2>&1
cat "$W/env.log"

echo "=== start server $(date +%T) ==="
cd /root
python3 -m sglang.launch_server \
  --model-path "$MODEL" --served-model-name "$NAME" \
  --host 127.0.0.1 --port $PORT --trust-remote-code \
  --attention-backend triton --mem-fraction-static 0.85 \
  --max-running-requests 8 --context-length $CTX --disable-cuda-graph \
  > "$W/server.log" 2>&1 &
SPID=$!
OK=0
for t in $(seq 1 180); do
  kill -0 $SPID 2>/dev/null || { echo "SERVER_EXITED at $((t*10))s"; break; }
  curl -sf --noproxy '*' -m 5 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && { OK=1; echo "SERVER_UP after $((t*10))s $(date +%T)"; break; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "SERVER_NOT_UP"; tail -40 "$W/server.log"; exit 1; }

echo "=== smoke.log $(date +%T) ==="
{
  echo "== /health"
  curl -sS -f --noproxy '*' -m 60 -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:$PORT/health
  echo "== /v1/models"
  curl -sS -f --noproxy '*' -m 60 http://127.0.0.1:$PORT/v1/models; echo
  echo "== 单轮"
  curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍国产异构加速器。\"}],\"temperature\":0,\"max_tokens\":128}"; echo
  echo "== 多轮"
  curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"我叫小海。\"},{\"role\":\"assistant\",\"content\":\"你好，小海！\"},{\"role\":\"user\",\"content\":\"我叫什么名字？\"}],\"temperature\":0,\"max_tokens\":128}"; echo
  echo "== 流式"
  curl -sS -f --noproxy '*' -m 300 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍国产异构加速器。\"}],\"temperature\":0,\"max_tokens\":64,\"stream\":true}" | tail -n 3
} > "$W/smoke.log" 2>&1
tail -6 "$W/smoke.log"

echo "=== bench $(date +%T) ==="
BENCH=sglang.benchmark.serving
python3 -c "import $BENCH" 2>/dev/null || BENCH=sglang.bench_serving
echo "bench module: $BENCH"
mkdir -p "$W/bench"; rm -f "$W"/bench/round*.jsonl
for ILEN in 4096 16384; do
  for C in 1 8 64; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG $(date +%T)"
      timeout 1800 python3 -m "$BENCH" \
        --backend sglang --host 127.0.0.1 --port $PORT \
        --model "$NAME" --tokenizer "$MODEL" \
        --dataset-name random-ids \
        --random-input-len $ILEN --random-output-len 1024 --random-range-ratio 1 \
        --max-concurrency $C --num-prompts $N --seed 1 \
        --output-file "$W/bench/round${R}.jsonl" > "$W/bench/${TAG}.log" 2>&1 \
        || echo "[bench] FAIL $TAG"
    done
  done
done

echo "=== summary $(date +%T) ==="
python3 - <<'PY' > "$W/bench/summary.md" 2>&1
import json
cols = ["random_input_len","random_output_len","max_concurrency","completed","duration",
        "request_throughput","output_throughput","total_throughput","mean_ttft_ms","p99_ttft_ms",
        "mean_tpot_ms","p99_tpot_ms","mean_e2e_latency_ms"]
print("| " + " | ".join(cols) + " |")
print("|" + " --- |"*len(cols))
try:
    for line in open("/work/bench/round2.jsonl", encoding="utf-8"):
        line=line.strip()
        if not line: continue
        r = json.loads(line)
        print("| " + " | ".join(("%.2f" % r[c]) if isinstance(r.get(c), float) else str(r.get(c,"-")) for c in cols) + " |")
except FileNotFoundError:
    print("no round2.jsonl")
PY
cat "$W/bench/summary.md"
kill $SPID 2>/dev/null
echo "=== DONE inner $(date +%T) ==="
EOS

echo "== run =="
$PD run --rm --ipc=host --device /dev/kfd --device /dev/mkfd --device /dev/dri --privileged \
  --security-opt seccomp=unconfined \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/MiMo:ro -v "$WORK":/work \
  -v /tmp/inner.sh:/inner.sh:ro "$IMG" bash /inner.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE outer $(date +%T) =="
