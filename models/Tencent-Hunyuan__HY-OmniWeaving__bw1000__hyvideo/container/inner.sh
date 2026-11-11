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
# 只应用与本条目源码匹配的补丁：先 dry-run，不匹配就显式报错并跳过（避免拷错补丁目录时误判）
for p in /work/repo/patches/*.patch; do
  [ -e "$p" ] || continue
  if patch -p1 --dry-run -d /opt/omni-src < "$p" >/dev/null 2>&1; then
    echo "  apply $(basename "$p")"; patch -p1 -d /opt/omni-src < "$p" >/dev/null || exit 5
  else
    echo "  [skip] $(basename "$p") 与本源码不匹配（dry-run 失败）"
  fi
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
# 【多卡要点】generate.py 用 WORLD_SIZE 做序列并行，torchrun 会设 WORLD_SIZE/RANK/LOCAL_RANK。
# 但容器里若残留 HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES（Slurm 只给 1 张卡的可见性），
# 其余 rank 会越界 → build_mesh 失败。多卡时必须 unset 让 torch 看到 cgroup 允许的全部卡。
# PROBE_FRESH=1：探针要真的生成（否则 PROBE_ONLY=1 会因为首条已存在而被跳过）
if [ "${PROBE_FRESH:-0}" = "1" ]; then
  export OUTDIR_OVERRIDE=official-eval/probe-outs
  mkdir -p "$OUTDIR_OVERRIDE"
  echo "[omni] PROBE_FRESH=1 → 输出到 $OUTDIR_OVERRIDE（不影响正式产物）"
fi
if [ "${NPROC:-1}" -gt 1 ]; then
  echo "[omni] 多卡模式：NPROC=$NPROC，unset 可见性变量"
  OUTDIR=official-eval/outputs PROBE_ONLY="${PROBE_ONLY:-0}" NPROC="$NPROC" \
    COND_FILE=conditions/omniweaving.jsonl \
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    python3 evaluate_official.py 2>&1 | tail -40
else
  OUTDIR=official-eval/outputs PROBE_ONLY="${PROBE_ONLY:-0}" NPROC="${NPROC:-1}" \
    COND_FILE=conditions/omniweaving.jsonl OMNI_SRC=/opt/omni-src MODEL_PATH=/models/Omni \
    env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES \
    python3 run_missing.py 2>&1 | tail -40
fi

echo '--- 产物规格校验（帧数/fps/分辨率）---'
RUN_TAG="${SLURM_JOB_ID:-x}" python3 - <<'PYEOF'
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
import os as _os
json.dump(stats, open(f"stats-{_os.environ.get('RUN_TAG','x')}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(stats, ensure_ascii=False))
PYEOF
echo '[omni] done'
