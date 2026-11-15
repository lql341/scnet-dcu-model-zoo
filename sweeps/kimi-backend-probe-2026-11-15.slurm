#!/bin/bash -l
# 1 卡只读探测：Kimi 16K/c64 崩溃的下一枪该打哪里（0 卡时决策用）
#   ① 0.20 的 MLA 为什么只有一个候选 backend（TRITON_MLA），能否换成 FLASHMLA
#   ② 0.21 的 vllm_hcu 在加载 3D 权重时无条件 .t()，有无开关绕开
# 不加载模型、不起服务，纯读源码 + 问枚举。
# 用法：在仓库根目录 sbatch sweeps/kimi-backend-probe-2026-11-15.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J kimi-bkprobe
#SBATCH -o /public/home/luql/work/kimi-bkprobe/probe_%j.out

set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
DIR=/public/home/luql/work/kimi-bkprobe
mkdir -p "$DIR"
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
MODEL_HOST=/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib'

cp "$REPO/sweeps/kimi-backend-probe.sh" "$DIR/probe.sh"
chmod a+r "$DIR/probe.sh"

log() { echo "[probe $(date +%H:%M:%S)] $*"; }
# 作业退出前必须删掉 podman 存储，否则镜像解包留在节点 /tmp，谁也删不掉（见 gotchas #40）
cleanup_podman() {
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_podman EXIT

echo "[probe] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"

run_image() {
  local tag="$1" img="$2" tarball="$3"
  log "===== $tag ====="
  log "load image"
  gunzip -c "$tarball" | $PD load >"$DIR/load-$tag-$JID.log" 2>&1 \
    || { log "FAIL load $tag"; tail -5 "$DIR/load-$tag-$JID.log"; return 1; }

  $PD run --rm --name "probe-$tag-$JID" --network=none --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$DIR":/gate \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH="$LIBPATH" \
    -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$img" /gate/probe.sh "$tag" 2>&1 | tee "$DIR/console-$tag-$JID.log"

  $PD rmi -f "$img" >/dev/null 2>&1
}

run_image vllm020 \
  "image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0" \
  "$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"
run_image vllm021 \
  "image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1" \
  "$IMGDIR/vllm0.21.0-ubuntu22.04-dtk2604-py3.10_v1.tar.gz"

log "done. 结果：$DIR/{vllm020,vllm021}.md"
