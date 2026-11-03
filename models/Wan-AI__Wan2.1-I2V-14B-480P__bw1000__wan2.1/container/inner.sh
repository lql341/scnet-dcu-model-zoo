#!/bin/bash
# Wan2.1 容器内步骤（独立脚本，避免 podman run -c 里的引号/$ 展开问题）
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal

echo "--- 镜像内版本 ---"
python3 -c 'import torch, transformers, torchvision; print("torch", torch.__version__, "| transformers", transformers.__version__, "| torchvision", torchvision.__version__)'

echo '--- 隔离安装纯 Python 依赖（--no-deps，cp310 轮子）---'
n_ok=0; n_fail=0
for w in /wheels/*.whl; do
  if pip install -q --no-deps --target /opt/wan-libs "$w" 2>/dev/null; then n_ok=$((n_ok+1)); else n_fail=$((n_fail+1)); echo "  [warn] fail: $(basename "$w")"; fi
done
echo "已装 $n_ok 个，失败 $n_fail 个"

# diffusers + hub 复用 Qwen-Image 那套（0.41.0 + 1.33.0）：
# wan 的 fm_solvers 只用到 ConfigMixin / register_to_config / scheduling_utils / randn_tensor 等稳定 API
mkdir -p /opt/qi-libs
for w in /wheels-qi/*.whl; do
  [ -e "$w" ] || continue
  pip install -q --no-deps --target /opt/qi-libs "$w" 2>/dev/null && echo "  ok(qi): $(basename "$w")" || echo "  [warn] fail(qi): $(basename "$w")"
done

# 官方代码只读挂载，容器内拷一份再打 patch（评测不需要 prompt 扩写，惰性导入 dashscope）
rm -rf /opt/wan-src && cp -r /wan-src /opt/wan-src
for p in /work/repo/patches/*.patch; do
  [ -e "$p" ] || continue
  echo "[wan] apply $(basename "$p")"
  patch -p1 -d /opt/wan-src < "$p" || exit 5
done
export WAN_SRC=/opt/wan-src
export PYTHONPATH=/opt/wan-libs:/opt/qi-libs:/opt/wan-src:${PYTHONPATH:-}
echo '--- 导入校验 ---'
python3 - <<'PYCHK'
import importlib, sys
bad = []
for m in ("torch", "torchvision", "diffusers", "einops", "ftfy", "regex", "imageio", "PIL",
          "transformers", "easydict", "tqdm"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        bad.append(m); print(f"  {m:<12} FAIL {type(e).__name__}: {e}")
sys.exit(3 if bad else 0)
PYCHK
if [ $? -eq 3 ]; then echo '[wan] 依赖不全，提前退出'; exit 3; fi
python3 -c 'import wan; from wan.configs import WAN_CONFIGS; print("wan import OK; tasks:", list(WAN_CONFIGS)[:6])' || exit 4
python3 -c 'import flash_attn' 2>/dev/null && echo "  flash_attn: 存在" || echo "  flash_attn: 缺失 → 注意力回落 torch SDPA（预期行为）"

echo '--- 官方条件评测（1 卡 1 进程 × SHARD_TOTAL）---'
cd /work/repo
mkdir -p outputs logs/shard
TOTAL=${SHARD_TOTAL:-1}
PROBE=${PROBE_ONLY:-0}
pids=""
for i in $(seq 0 $((TOTAL-1))); do
  # 每进程只看见一张卡；不要 unset HIP_VISIBLE_DEVICES（会越界枚举宿主的 8 张卡）
  HIP_VISIBLE_DEVICES=$i SHARD_INDEX=$i SHARD_TOTAL=$TOTAL \
    python3 evaluate_official.py > "logs/shard/shard$i.log" 2>&1 &
  pids="$pids $!"
  echo "[wan] shard $i -> pid $! (HIP_VISIBLE_DEVICES=$i)"
  [ "$PROBE" = "1" ] && break
done
fail=0
for pid in $pids; do wait "$pid" || fail=$((fail+1)); done
echo "[wan] 分片结束 fail=$fail"
for f in logs/shard/shard*.log; do [ -e "$f" ] && { echo "--- $f"; tail -3 "$f"; }; done

python3 - <<'PYMERGE'
import csv, glob, os
rows = []
for f in sorted(glob.glob("反馈表-shard*.csv")):
    with open(f, encoding="utf-8") as fh:
        rows += list(csv.DictReader(fh))
if rows:
    with open("反馈表.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    ok = sum(1 for r in rows if r["status"] == "成功")
    print("[wan] 合并 %d 行，成功 %d" % (len(rows), ok))
PYMERGE

python3 - <<'PYSTATS'
import glob, json, os
try:
    import imageio.v2 as iio
except Exception as e:
    print("[wan] imageio 不可用，跳过规格校验:", e); raise SystemExit(0)
files = sorted(glob.glob("outputs/*.mp4"))
bad, durs = [], []
for f in files:
    try:
        r = iio.get_reader(f); meta = r.get_meta_data(); r.close()
        fps = round(float(meta.get("fps", 0)), 2); dur = float(meta.get("duration", 0))
        if fps != 16 or abs(dur - 81 / 16) > 0.2:
            bad.append({"file": os.path.basename(f), "fps": fps, "dur": round(dur, 2)})
        durs.append(dur)
    except Exception as e:
        bad.append({"file": os.path.basename(f), "error": str(e)[:100]})
stats = {"files": len(files),
         "bytes_total_MB": round(sum(os.path.getsize(f) for f in files) / 1e6, 1),
         "dur_s": {"min": min(durs) if durs else None, "max": max(durs) if durs else None},
         "spec_violation_count": len(bad), "spec_violations": bad[:10]}
json.dump(stats, open("stats.json", "w"), ensure_ascii=False, indent=1)
print("[wan] 规格校验:", json.dumps(stats, ensure_ascii=False)[:400])
PYSTATS
echo '[wan] shard done'
