#!/bin/bash
# OmniWeaving 容器内步骤：装隔离依赖 → 打 DCU 兼容补丁 → 逐条跑官方条件
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo '--- 镜像内版本 ---'
python3 -c 'import torch, torchvision; print("torch", torch.__version__, "| torchvision", torchvision.__version__)'

echo '--- 隔离安装依赖（--no-deps，cp310 轮子）---'
n=0; f=0
for w in /wheels/*.whl; do
  if pip install -q --no-deps --target /opt/omni-libs "$w" 2>/dev/null; then n=$((n+1)); else f=$((f+1)); echo "  [warn] fail: $(basename "$w")"; fi
done
echo "已装 $n，失败 $f"

echo '--- 复制源码并打补丁（源码只读挂载）---'
rm -rf /opt/omni-src && cp -r /omni-src /opt/omni-src
for p in /work/repo/patches/*.patch; do
  [ -e "$p" ] || continue
  echo "  apply $(basename "$p")"; patch -p1 -d /opt/omni-src < "$p" || exit 5
done
export PYTHONPATH=/opt/omni-libs:/opt/omni-src:${PYTHONPATH:-}
export OMNI_SRC=/opt/omni-src

python3 - <<'PYCHK'
import importlib
for m in ("torch", "torchvision", "transformers", "diffusers", "huggingface_hub", "peft",
          "einops", "loguru", "imageio", "omegaconf", "qwen_vl_utils", "decord", "cv2"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<16} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<16} FAIL {type(e).__name__}: {e}")
PYCHK
python3 -c 'import hyvideo; print("hyvideo import OK")' || exit 4

echo '--- 官方条件评测 ---'
cd /work/repo
mkdir -p official-eval/outputs logs/shard
OUTDIR=official-eval/outputs PROBE_ONLY="${PROBE_ONLY:-0}" NPROC="${NPROC:-1}" \
  COND_FILE=conditions/omniweaving.jsonl python3 evaluate_official.py 2>&1 | tail -30

echo '--- 产物规格校验（帧数/fps/分辨率）---'
python3 - <<'PYEOF'
import glob, json, os
try:
    import imageio.v2 as iio
except Exception as e:
    print("[omni] imageio 不可用，跳过:", e); raise SystemExit(0)
files = sorted(glob.glob("official-eval/outputs/*.mp4"))
bad, durs = [], []
for f in files:
    try:
        r = iio.get_reader(f); meta = r.get_meta_data(); r.close()
        fps = round(float(meta.get("fps", 0)), 2); dur = float(meta.get("duration", 0))
        if fps != 16 or abs(dur - 81 / 16) > 0.25:
            bad.append({"file": os.path.basename(f), "fps": fps, "dur": round(dur, 2)})
        durs.append(dur)
    except Exception as e:
        bad.append({"file": os.path.basename(f), "error": str(e)[:80]})
stats = {"files": len(files),
         "bytes_total_MB": round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
         "dur_s": {"min": min(durs) if durs else None, "max": max(durs) if durs else None},
         "spec_violation_count": len(bad), "violations": bad[:5]}
json.dump(stats, open("stats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False))
PYEOF
echo '[omni] done'
