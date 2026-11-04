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
# 【坑】WORK 不能等于提交目录（REPO），否则 `cp -rf REPO/. WORK/repo/` 会递归拷自己，
# 导致容器里 /work/repo 缺文件（实测踩过：python3: can't open file '/work/repo/evaluate_official.py'）
WORK="${WORK:-/public/home/scnethr2601a10/work/m3run}"
[ "$WORK" = "$REPO" ] && { echo "[host] WORK 不能等于 REPO，请另指定 WORK"; exit 5; }
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
# 压缩 12.27 GB，但**镜像里含作者遗留的 14 GB core dump**（位于 /tmp/music3-hygon），
# 实测解包需 >56 GB，56 G 的节点直接 no space left on device → 门槛设 80 G
[ "$FREE" -lt 80 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

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
