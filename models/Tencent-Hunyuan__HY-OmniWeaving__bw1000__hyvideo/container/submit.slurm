#!/bin/bash -l
# 一键复现：HY-OmniWeaving × BW1000（官方 hyvideo + 序列并行）
#   sbatch container/submit.slurm                       # 官方 60 条（NPROC 卡协同，逐条串行）
#   PROBE_ONLY=2 sbatch container/submit.slurm          # 只跑 2 条
# 注意：本集群 sbatch 命令行的 --gres 覆盖无效，卡数只能改文件头
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -J omniweaving
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/omnirun}"
[ "$WORK" = "$REPO" ] && { echo "[host] WORK 不能等于 REPO"; exit 5; }
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
NCARD=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -m1 -oE 'gres/dcu=[0-9]+' | cut -d= -f2)
[ -z "${NCARD:-}" ] && NCARD=1
echo "[host] node=$(hostname) tmp_free=${FREE}G 卡数=$NCARD"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

$PD image exists "$IMAGE" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/pull.log; exit 1; }
mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work \
  -v "$OMNI_SRC_HOST":/omni-src:ro -v "$WHEELS_HOST":/wheels:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  -e PROBE_ONLY="${PROBE_ONLY:-0}" -e NPROC="$NCARD" -e MODEL_PATH="$MODEL_PATH" \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] outputs=$(ls "$WORK/repo/official-eval/outputs" 2>/dev/null | wc -l) DONE $(date +%T)"
