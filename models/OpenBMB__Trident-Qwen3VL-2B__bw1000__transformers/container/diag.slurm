#!/bin/bash -l
# 一次性诊断作业：图文路径的 reshape 不匹配出在哪一层（预处理 or 模型代码）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 01:00:00
#SBATCH -J trident-diag
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"; source "$REPO/env.sh"
WORK="${WORK:-/public/home/luql/work/trident-diag}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$; JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
mkdir -p "$XDG_RUNTIME_DIR" "$WORK" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
case "$HOST_IMAGE_TAR" in
  *.gz) gunzip -c "$HOST_IMAGE_TAR" | $PD load >/tmp/load-$JID.log 2>&1 ;;
  *)    $PD load -i "$HOST_IMAGE_TAR" >/tmp/load-$JID.log 2>&1 ;;
esac
if ! $PD image exists "$IMAGE" 2>/dev/null; then
  echo "[host] 归档 load 失败，尾部日志："; tail -5 /tmp/load-$JID.log; exit 1
fi
mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work -v "$WHEELS_HOST":/wheels:ro \
  -e MODEL_PATH="$MODEL_PATH" -e PYLIBS="$PYLIBS" \
  "$IMAGE" -c '
set -uo pipefail
cd /work/repo
echo "===== A：镜像自带 transformers + hub ====="
python3 - <<PYA
import transformers, huggingface_hub
print("transformers", transformers.__version__, "| hub", huggingface_hub.__version__)
PYA
python3 diag_images.py "$MODEL_PATH" 2>&1 | grep -E "^\[diag\]|RuntimeError|ImportError" | head -20
echo "===== B：隔离装 transformers 4.57.1 + hub 0.34.0 ====="
mkdir -p $PYLIBS
pip install -q --no-deps --upgrade --target $PYLIBS /wheels/huggingface_hub-0.34.0-py3-none-any.whl
pip install -q --no-deps --upgrade --target $PYLIBS /wheels/transformers-4.57.1-py3-none-any.whl
pip install -q --no-deps --upgrade --target $PYLIBS /wheels/qwen_vl_utils-0.0.8-py3-none-any.whl
PYTHONPATH=$PYLIBS:${PYTHONPATH:-} python3 diag_images.py "$MODEL_PATH" 2>&1 | grep -E "^\[diag\]|RuntimeError|ImportError" | head -20
' 2>&1 | grep -viE "subuid|rootless|level=warning|WARNING: Running pip"
