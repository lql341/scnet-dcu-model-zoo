#!/bin/bash -l
# AuK-Flash × BW1000 探针：容器内逻辑放在 container/inner.sh，避免嵌套引号问题
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
# AuK-Flash 是 nfe=4 的蒸馏版，单条预计 <1 s；留 3 h 墙钟给加载与意外；支持产物续跑
#SBATCH -t 03:00:00
#SBATCH -J aukflash-probe
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/auk-flash}"
ASSETS=/public/home/scnethr2601a10/work/official-t34
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 70 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then $PD load -i "$HOST_IMAGE_TAR" >/tmp/load.log 2>&1; fi
$PD image exists "$IMAGE" 2>/dev/null || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/load.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/load.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$HOST_ENCODER_PATH":$ENCODER_PATH:ro \
  -v "$WORK":/work -v "$ASSETS":/assets:ro -v "$WHEELS_HOST":/wheels:ro -v "$AUK_SRC_HOST":/auk-src:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] DONE $(date +%T)"
