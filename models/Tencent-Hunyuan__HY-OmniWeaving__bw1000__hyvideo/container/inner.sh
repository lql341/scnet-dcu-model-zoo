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
# 【多卡要点】实测：本集群即使申请 dcu:4，Slurm 也可能只把 1 张卡写进
# HIP_VISIBLE_DEVICES（本作业实测 host 侧 CUDA_VISIBLE_DEVICES=0 / HIP_VISIBLE_DEVICES=0，
# 而 SLURM_JOB_GPUS 只给了个数字 4）。此时"unset 可见性变量"仍拿不到 4 张卡，
# 4 个 rank 会在 set_device 处报 `HIP error: invalid device ordinal`（rank1/2/3 全挂）。
# 已知可行的做法（Kimi × vLLM TP=4 已验证）：显式写满 0..N-1。
VIS_LIST=$(seq -s, 0 $(( ${NPROC:-1} - 1 )))
echo "[omni] 多卡模式：NPROC=$NPROC，显式可见性 HIP_VISIBLE_DEVICES=$VIS_LIST"
OUTDIR="${OUTDIR_OVERRIDE:-official-eval/outputs}" PROBE_ONLY="${PROBE_ONLY:-0}" NPROC="$NPROC" \
  COND_FILE=conditions/omniweaving.jsonl OMNI_SRC=/opt/omni-src MODEL_PATH=/models/Omni \
  HIP_VISIBLE_DEVICES="$VIS_LIST" CUDA_VISIBLE_DEVICES="$VIS_LIST" \
  python3 - <<'PYCHK' 2>&1 | tail -40
import os, subprocess, sys
import torch
n = torch.cuda.device_count()
want = int(os.environ.get("NPROC", "1"))
print(f"[omni] 容器内可见 GPU 数={n}（期望 {want}）", flush=True)
if n < want:
    print("[omni] 设备数不足，终止（检查 Slurm 的 HIP_VISIBLE_DEVICES 与 podman --device）", flush=True)
    sys.exit(9)
sys.exit(subprocess.call([sys.executable, "run_missing.py"]))
PYCHK
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
