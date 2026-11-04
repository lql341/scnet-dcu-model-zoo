#!/bin/bash -l
# 一键复现：MiniMax-Music3 × BW1000（镜像自带 Hygon SGLang + BW 适配 harness）
#   sbatch container/submit.slurm                          # 官方 60 条
#   PROBE_ONLY=2 sbatch container/submit.slurm             # 只跑 2 条验管线
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -J music3
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/zoo-music3}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
# 12.27 GB 镜像解包需 ~35-45 GB
[ "$FREE" -lt 55 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

$PD image exists "$IMAGE" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/pull.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work \
  -v "$HARNESS_HOST":/work/harness -v "$CONDITIONS_HOST":/work/official-conds/music3.jsonl:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  -e PROBE_ONLY="${PROBE_ONLY:-0}" -e PORT="$PORT" \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] outputs=$(ls "$WORK/repo/official-eval/outputs" 2>/dev/null | wc -l)  DONE $(date +%T)"
