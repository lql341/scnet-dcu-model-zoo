#!/bin/bash -l
# Trident-Qwen3VL-2B × BW1000：自建 OpenAI 兼容 embedding 服务（单卡）
#   sbatch container/submit.slurm
#   ROUNDS=1 sbatch --export=ALL container/submit.slurm   # 压测少跑一轮（自检用）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J trident-emb
set -uo pipefail

REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交（找不到 $REPO/env.sh）"; exit 1; }
source "$REPO/env.sh"

# WORK 必须与 REPO 分开：相等时 cp 会把目录拷进自己，GNU cp 直接报错退出（静默失败），
# 运行时就找不到 serve_trident.py（实测 2026-11-15 第一次跑踩到）。
WORK="${WORK:-/public/home/luql/work/trident-run}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

mkdir -p "$XDG_RUNTIME_DIR" "$WORK" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

# 先从归档 tar.gz 加载（离线、可复现），失败再退回 Harbor 拉取
if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then
  case "$HOST_IMAGE_TAR" in
    *.gz) gunzip -c "$HOST_IMAGE_TAR" | $PD load >/tmp/load-$JID.log 2>&1 ;;
    *)    $PD load -i "$HOST_IMAGE_TAR" >/tmp/load-$JID.log 2>&1 ;;
  esac
fi
$PD image exists "$IMAGE" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull-$JID.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/load-$JID.log /tmp/pull-$JID.log 2>/dev/null; exit 1; }

mkdir -p "$WORK/repo"
if [ "$(cd "$REPO" && pwd -P)" = "$(cd "$WORK/repo" && pwd -P)" ]; then
  echo "[host] WORK/repo 与 REPO 相同，跳过拷贝"
else
  cp -rf "$REPO"/. "$WORK/repo"/ || { echo "[host] 拷贝仓库失败"; exit 1; }
fi

echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work -v "$WHEELS_HOST":/wheels:ro \
  -v "$WHEELS_EXTRA_HOST":/wheels-extra:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  -e MODEL_PATH="$MODEL_PATH" -e MODEL_ID="$MODEL_ID" -e PORT="$PORT" \
  -e EMB_BATCH="$EMB_BATCH" -e EMB_MAX_TOKENS="$EMB_MAX_TOKENS" \
  -e PYLIBS="$PYLIBS" -e ROUNDS="${ROUNDS:-2}" \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
RC=${PIPESTATUS[0]}
echo "[host] DONE $(date +%T) rc=$RC"
[ -f "$WORK/repo/bench/summary.md" ] && { echo "[host] --- bench/summary.md"; cat "$WORK/repo/bench/summary.md"; }
exit $RC
