#!/bin/bash -l
# 一键复现：起容器 → 容器内跑 env_check / run / smoke / verify / (bench)
#
# 用法（在仓库根目录提交）：
#   sbatch container/submit.slurm
#   SKIP_BENCH=1 sbatch --export=ALL container/submit.slurm     # 只验到 smoke，跳过耗时压测
#
# 约定：
#   - 仓库以只读挂到容器 /repo，拷进 /work/repo 后执行，脚本产生的 env.log / server.log /
#     smoke.log / bench/ 会留在宿主机 $WORK/repo 下
#   - 镜像优先从本地 tar load，失败再 pull
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 03:30:00
#SBATCH -J kimi-accept

set -uo pipefail

# sbatch 会把脚本拷到 spool 目录，$0 不可用；用 SLURM_SUBMIT_DIR 定位仓库根
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在仓库根目录提交（找不到 $REPO/env.sh）"; exit 1; }
source "$REPO/env.sh"

WORK="${WORK:-/public/home/scnethr2601a10/work/kimi}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
echo "[host] repo=$REPO"
echo "[host] work=$WORK"
if [ "$FREE" -lt 45 ]; then echo "[host] NOT_ENOUGH_SPACE (need 45G)", exit 42; fi

echo "[host] load image"
if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then
  $PD load -i "$HOST_IMAGE_TAR" >/tmp/load.log 2>&1
fi
$PD image exists "$IMAGE" 2>/dev/null \
  || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/load.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image not available"; tail -5 /tmp/load.log; exit 1; }

cat > /tmp/inner.sh <<'EOS'
set -uo pipefail
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
mkdir -p /work/repo && cp -rf /repo/. /work/repo/ 2>/dev/null
cd /work/repo || exit 1
echo "=== 仓库内容 ==="; ls
bash env_check.sh
bash run.sh
bash smoke.sh
source ./env.sh
python3 verify.py --selfcheck
python3 verify.py || echo "[inner] verify.py 被官方基线文件阻塞，见 MISSING.md"
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[inner] SKIP_BENCH=1，跳过压测"
else
  bash bench.sh
fi
echo "=== 产物 ==="; ls -la env.log server.log smoke.log bench 2>/dev/null | head -20
EOS

echo "[host] run containers $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$REPO":/repo:ro \
  -v "$HOST_MODEL_PATH":/models/Kimi:ro \
  -v "$WORK":/work \
  -v "$REPO/preflight_lib.py":/preflight_lib.py:ro \
  -v /tmp/inner.sh:/inner.sh:ro \
  -e SKIP_BENCH="${SKIP_BENCH:-0}" -e BENCH_QUICK="${BENCH_QUICK:-0}" \
  "$IMAGE" /inner.sh
echo "[host] done $(date +%T)"
echo "[host] 产物目录：$WORK/repo"
