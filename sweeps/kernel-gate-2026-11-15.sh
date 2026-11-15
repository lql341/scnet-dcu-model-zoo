#!/bin/bash -l
# 0 卡时闸门（HANDOFF §3.1）——只回答四个问题，不加载模型、不起服务
#   ① vLLM 有哪些 attention backend   ② aiter 是否存在
#   ③ hipBLASLt 的 fp8 支持           ④ Kimi-Linear 走哪套 kernel
# 两个镜像各跑一遍：vllm0.20 / vllm0.21
# 这只是 1 卡作业（DCU 是分配粒度），但计算量≈0：纯 import + grep。
# 用法：在仓库根目录 sbatch sweeps/kernel-gate-2026-11-15.sh
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J kernel-gate
#SBATCH -o /public/home/luql/work/kernel-gate/gate_%j.out

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
GATE_DIR=/public/home/luql/work/kernel-gate
mkdir -p "$GATE_DIR"
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib'

[ -f "$REPO/sweeps/kernel-gate-probe.sh" ] || { echo "[gate] 找不到 kernel-gate-probe.sh"; exit 1; }
cp "$REPO/sweeps/kernel-gate-probe.sh" "$GATE_DIR/gate_probe.sh"
chmod a+r "$GATE_DIR/gate_probe.sh"

log() { echo "[gate $(date +%H:%M:%S)] $*"; }
echo "[gate] node=$(hostname) dcu=$(ls /dev/kfd 2>/dev/null) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"

run_image() {
  local tag="$1" img="$2" tarball="$3"
  log "===== $tag : $img ====="
  log "load image $tag"
  gunzip -c "$tarball" | $PD load >"$GATE_DIR/load-$tag-$JID.log" 2>&1 \
    || { log "FAIL load $tag"; tail -5 "$GATE_DIR/load-$tag-$JID.log"; return 1; }

  # LD_LIBRARY_PATH 必须传进去：漏掉时 import torch 报 libgalaxyhip.so.5，
  # 看起来像镜像坏了，其实是探测脚本自己的环境不全（与 gotchas #8 同类）。
  $PD run --rm --name "gate-$tag-$JID" --network=none --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$GATE_DIR":/gate \
    -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH="$LIBPATH" \
    -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$img" /gate/gate_probe.sh "$tag" 2>&1 | tee "$GATE_DIR/console-$tag-$JID.log"

  $PD rmi -f "$img" >/dev/null 2>&1
}

run_image vllm020 \
  "image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0" \
  "$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"
run_image vllm021 \
  "image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1" \
  "$IMGDIR/vllm0.21.0-ubuntu22.04-dtk2604-py3.10_v1.tar.gz"

log "done. results in $GATE_DIR/{vllm020,vllm021}.md"
