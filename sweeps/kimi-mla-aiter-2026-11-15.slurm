#!/bin/bash -l
# 1 卡只读：为什么 0.20 上 Kimi 的 MLA 候选集只有 TRITON_MLA，能否切到 FLASHMLA/AITER
# 用法：在仓库根目录 sbatch sweeps/kimi-mla-aiter-2026-11-15.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J kimi-mlaprobe
#SBATCH -o /public/home/luql/work/kimi-mlaprobe/probe_%j.out

set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
DIR=/public/home/luql/work/kimi-mlaprobe
mkdir -p "$DIR"
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib'

cp "$REPO/sweeps/kimi-mla-aiter-probe.sh" "$DIR/probe.sh"; chmod a+r "$DIR/probe.sh"
IMG="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
TAR="$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"

# 作业退出前必须删掉 podman 存储，否则镜像解包留在节点 /tmp，谁也删不掉（见 gotchas #40）
cleanup_podman() {
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_podman EXIT

echo "[mlaprobe] node=$(hostname)"
gunzip -c "$TAR" | $PD load >"$DIR/load-$JID.log" 2>&1 || { echo "FAIL load"; tail -5 "$DIR/load-$JID.log"; exit 1; }
$PD run --rm --name "mlap-$JID" --network=none --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$DIR":/gate \
  -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH="$LIBPATH" \
  -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
  "$IMG" /gate/probe.sh 2>&1 | tee "$DIR/console-$JID.log"
$PD rmi -f "$IMG" >/dev/null 2>&1
echo "[mlaprobe] done"
