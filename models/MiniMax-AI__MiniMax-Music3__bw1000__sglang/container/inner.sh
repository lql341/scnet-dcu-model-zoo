#!/bin/bash
# Music3 容器内步骤：起 harness 服务 → 跑官方 60 条 → 收产物
set -uo pipefail
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo '--- 镜像内版本 ---'
python3 -c 'import torch, transformers, vllm; print("torch", torch.__version__, "| transformers", transformers.__version__, "| vllm", vllm.__version__)' 2>&1 | tail -2
hy-smi 2>/dev/null | head -6 || true

echo '--- 准备 harness（镜像作者的工作区，已归档；先清掉悬空符号链接）---'
cd /work/harness || { echo '[m3] 缺 harness'; exit 2; }
rm -rf runtime logs .venv server.pid server.log __pycache__ music3_hygon/__pycache__ 2>/dev/null
mkdir -p runtime logs
source ./env.sh
export MODEL_PATH=/models/Music3 RUNTIME_PATH=/work/harness/runtime PORT="${PORT:-8000}"

echo '--- env_check ---'
bash env_check.sh 2>&1 | tail -8 || true

echo '--- 起服务（run.sh：打兼容 patch → venv → 轮询 /health）---'
bash run.sh 2>&1 | tail -6 || { echo '[m3] 服务启动失败'; tail -25 server.log 2>/dev/null; exit 3; }

echo '--- 官方条件评测 ---'
cd /work/repo
mkdir -p official-eval/outputs
OUTDIR=official-eval/outputs PROBE_ONLY="${PROBE_ONLY:-0}" PORT="${PORT:-8000}" \
  COND_FILE=conditions/music3.jsonl python3 evaluate_official.py 2>&1 | tail -25

echo '--- 产物规格抽检 ---'
python3 - <<'PYEOF'
import glob, json, os, wave
files = sorted(glob.glob("official-eval/outputs/*.wav"))
bad, durs = [], []
for f in files:
    try:
        with wave.open(f) as w:
            sr, ch, dur = w.getframerate(), w.getnchannels(), w.getnframes() / w.getframerate()
        if sr != 32000 or ch != 2:
            bad.append({"file": os.path.basename(f), "sr": sr, "ch": ch})
        durs.append(round(dur, 2))
    except Exception as e:
        bad.append({"file": os.path.basename(f), "error": str(e)[:80]})
print(json.dumps({"files": len(files),
                  "bytes_total_MB": round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
                  "dur_s": {"min": min(durs) if durs else None, "max": max(durs) if durs else None},
                  "spec_violation_count": len(bad), "violations": bad[:5]}, ensure_ascii=False))
PYEOF
pkill -f serve.py 2>/dev/null || true
echo '[m3] done'
