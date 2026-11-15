#!/bin/bash -l
# P0-B 闸门：量化白名单 + fp8 能否真的做 GEMM（0 卡时）
# 两个镜像各跑一遍：0.20（RocmPlatform）与 0.21（HCUPlatform）
#SBATCH -p hx1hdnormal
#SBATCH --exclude=f11r1n04,f11r1n20,f11r1n17,f11r1n18,f11r1n01,f11r1n02,f11r2n13,f11r3n09
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J kimi-quantigate
#SBATCH -o /public/home/luql/work/kimi-quantigate/probe_%j.out
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
DIR=/public/home/luql/work/kimi-quantigate; mkdir -p "$DIR"
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID; mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
cleanup_podman() { $PD rmi -af >/dev/null 2>&1; $PD system prune -af >/dev/null 2>&1; rm -rf "/tmp/pr-$JID" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null; }
trap cleanup_podman EXIT
IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'
cp "$REPO/sweeps/kimi-quantigate-probe.sh" "$DIR/probe.sh"; chmod a+r "$DIR/probe.sh"
log() { echo "[qg $(date +%H:%M:%S)] $*"; }
echo "[qg] node=$(hostname) tmp_free=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')G"
[ "$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')" -ge 45 ] || { echo "NOT_ENOUGH_SPACE"; exit 42; }

run_img() {  # $1=tag $2=image $3=tarball
  local tag="$1" img="$2" tar="$3"
  log "===== $tag ====="
  gunzip -c "$tar" | $PD load >"$DIR/load-$tag-$JID.log" 2>&1 \
    || { log "FAIL load $tag"; tail -3 "$DIR/load-$tag-$JID.log"; return 1; }
  $PD run --rm --network=none --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$DIR":/gate \
    -e HIP_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH="$LIBPATH" \
    -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$img" /gate/probe.sh "$tag" 2>&1 | tee "$DIR/console-$tag-$JID.log"
  $PD rmi -f "$img" >/dev/null 2>&1
}

run_img vllm020 \
  "image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0" \
  "$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"

# 两个镜像顺序解包会撞磁盘，0.21 单独一发更稳
if [ "${QG_021:-0}" = "1" ]; then
  run_img vllm021 \
    "image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1" \
    "$IMGDIR/vllm0.21.0-ubuntu22.04-dtk2604-py3.10_v1.tar.gz"
fi
echo done
