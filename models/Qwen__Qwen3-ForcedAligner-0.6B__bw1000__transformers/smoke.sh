#!/usr/bin/env bash
# Smoke：单条对齐，确认 qwen_asr + 隔离 transformers 链路可用
set -euo pipefail
source ./env.sh
mkdir -p logs
exec > >(tee logs/smoke.log) 2>&1
PYTHONPATH=/opt/fa-libs python3 - <<'PY'
import os, torch, json
from qwen_asr import Qwen3ForcedAligner
a = os.path.join(os.environ.get("ASSET_ROOT", "/assets"),
                 "data/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac")
print("音频:", a, "| 存在:", os.path.exists(a))
m = Qwen3ForcedAligner.from_pretrained(os.environ["MODEL_PATH"], dtype=torch.bfloat16, device_map="cuda:0")
r = m.align(audio=a, text="HE HOPED THERE WOULD BE STEW FOR DINNER", language="English")
print("前 5 个词:", [(w.text, round(w.start_time,2), round(w.end_time,2)) for w in r[0][:5]])
PY
