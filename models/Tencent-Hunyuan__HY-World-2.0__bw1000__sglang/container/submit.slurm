#!/bin/bash -l
# HY-World-2.0（WorldMirror 2.0）× BW1000：官方 5 条 DTU 场景重建
#   复用 Harbor 社区镜像（内含 gsplat HIP 补丁 + harness），不做任何自建
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J hyworld
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
WORK="${WORK:-/public/home/luql/work/hyworld}"
IDS="${IDS:-dtu_scan1 dtu_scan4 dtu_scan9 dtu_scan10 dtu_scan11}"
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID=${SLURM_JOB_ID:-$$}
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G ids=$IDS"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMAGE" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull-$JID.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/pull-$JID.log; exit 1; }

# harness 拷到工作区（容器内 build_gsplat 要写 upstream/）
rm -rf "$WORK/harness" "$WORK"/out-*; mkdir -p "$WORK"
rsync -a "$HARNESS_HOST/" "$WORK/harness/" 2>/dev/null
chmod -R a+rwX "$WORK" 2>/dev/null
echo "[host] harness=$(du -sh "$WORK/harness" | cut -f1) run $(date +%T)"

LIB=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$WORK":/work -v "$HOST_MODEL_PATH":$MODEL_PATH:ro \
  -v "$MATERIALS_HOST":/evaluation-materials:ro \
  -e LIB="$LIB" -e IDS="$IDS" "$IMAGE" -c '
set -uo pipefail
export LD_LIBRARY_PATH="$LIB:${LD_LIBRARY_PATH:-}"
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/harness || exit 2
echo "--- build gsplat（镜像内已有编译缓存）---"
bash build_gsplat.sh 2>&1 | tail -8 || { echo "[hyw] gsplat 构建失败"; exit 3; }
bash env_check.sh 2>&1 | tail -10 || true
# run_conditions.py 里 skyseg.onnx 是相对 cwd 的路径
ln -sf assets/skyseg.onnx skyseg.onnx
source ./env.sh
export MODEL_PATH=/models/HY-World-2.0 HY_WORLD_EVALUATION_MATERIALS=/evaluation-materials
echo "--- 官方 5 条条件 ---"
python3 -u run_conditions.py --condition-file /evaluation-materials/formal/conditions/hy_world.jsonl \
  --ids $IDS --output "/work/out-$(date +%H%M%S)" 2>&1 | tail -30
echo "--- 结果 ---"
for f in /work/out-*/state.json; do python3 -c "
import json,sys
s=json.load(open(\"$f\"))
print(\"status:\", s.get(\"status\"))
for c in s.get(\"conditions\",[]): print(\" \", c[\"id\"], c[\"status\"], round(c.get(\"duration_seconds\") or 0), \"s\")
" 2>/dev/null; done
' 2>&1 | grep -viE "subuid|rootless|level=warning|^Copying|^Getting|UserWarning|setattr|_float_to_str"
echo "[host] DONE $(date +%T)"
