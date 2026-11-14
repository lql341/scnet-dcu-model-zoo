#!/bin/bash -l
# 一键复现：HY-OmniWeaving × BW1000（官方 hyvideo + 序列并行）
#   sbatch container/submit.slurm                       # 官方 60 条（NPROC 卡协同，逐条串行）
#   PROBE_ONLY=2 sbatch container/submit.slurm          # 只跑 2 条
# 注意：本集群 sbatch 命令行的 --gres 覆盖无效，卡数只能改文件头
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1          # 多卡 SP 在这套环境上 build_mesh 失败（设备可见性），先用单卡跑完
#SBATCH -c 16
#SBATCH -t 2-00:00:00
#SBATCH -J omniweaving
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
WORK="${WORK:-/public/home/luql/work/omnirun}"   # 2026-11-11 起用 luql 账号
[ "$WORK" = "$REPO" ] && { echo "[host] WORK 不能等于 REPO"; exit 5; }
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
# 卡数检测（三重兜底）：本集群实测 scontrol 的 gres/dcu 有时给出的是 1（与申请不符），
# 因此再用 SLURM_JOB_GPUS / HIP_VISIBLE_DEVICES 交叉校验，取三者最大值。
_n1=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -m1 -oE 'gres/dcu=[0-9]+' | cut -d= -f2)
_n2=$(echo "${SLURM_JOB_GPUS:-}" | awk -F, '{print NF}')
_n3=$(echo "${HIP_VISIBLE_DEVICES:-}" | awk -F, '{print NF}')
NCARD=$(printf '%s\n%s\n%s\n' "${_n1:-0}" "${_n2:-0}" "${_n3:-0}" | sort -n | tail -1)
# 记录配额与可见性实况（多卡问题的第一手证据，排查时不必再复跑探针）
scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -E "^(ReqTRES|AllocTRES)=" | sed 's/^/[host] /'
echo "[host] 可见性: SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "[host] node=$(hostname) 卡数=$NCARD"
echo "[host] 卡数检测: scontrol=${_n1:-?} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-?}(=$_n2) HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-?}(=$_n3) → NCARD=$NCARD"
[ "${NCARD:-0}" -lt 1 ] && NCARD=1
echo "[host] node=$(hostname) tmp_free=${FREE}G 卡数=$NCARD"
# SGLang 镜像解包 32 GB，podman 还要一份临时空间；45 G 实测会 no space（43 G 被拒）
[ "$FREE" -lt 50 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

# 先从归档 tar.gz 加载（离线、可控），失败再退回 Harbor 拉取
if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then
  case "$HOST_IMAGE_TAR" in
    *.gz) gunzip -c "$HOST_IMAGE_TAR" | $PD load >/tmp/load-$JID.log 2>&1 ;;
    *)    $PD load -i "$HOST_IMAGE_TAR" >/tmp/load-$JID.log 2>&1 ;;
  esac
fi
$PD image exists "$IMAGE" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/pull-$JID.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/pull-$JID.log; exit 1; }
mkdir -p "$WORK/repo"
# 【坑】cp -rf 在本机 NFS 上既不会可靠覆盖旧文件、也可能整片漏拷（实测漏了 run_missing.py）。
# 改用 rsync：能正确更新变更文件，同时保留 official-eval/outputs 里的已有产物。
if [ "${SKIP_SYNC:-0}" != "1" ]; then
  rsync -a --exclude 'official-eval/outputs' --exclude 'logs' "$REPO"/ "$WORK/repo"/ 2>/dev/null
else
  echo "[host] SKIP_SYNC=1：使用 $WORK/repo 里已准备好的文件"
fi
echo "[host] REPO=$REPO WORK=$WORK  run_missing.py=$(ls "$WORK/repo/run_missing.py" 2>/dev/null || echo 缺失)"
chmod -R a+rwX "$WORK/repo" 2>/dev/null
# 容器 rootless podman 的 root 映射成 subuid：**新建**文件可以，**覆盖**已存在的 666 文件在 NFS 上仍报
# Permission denied（实测 反馈表.csv / stats.json）。宿主先删掉，让容器新建。
rm -f "$WORK/repo/反馈表.csv" "$WORK/repo/stats.json" "$WORK/repo/eval_result.json" 2>/dev/null
echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work \
  -v "$OMNI_SRC_HOST":/omni-src:ro -v "$WHEELS_HOST":/wheels:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  -e PROBE_ONLY="${PROBE_ONLY:-0}" -e PROBE_FRESH="${PROBE_FRESH:-0}" -e OUTDIR_OVERRIDE="${OUTDIR_OVERRIDE:-}" -e NPROC="$NCARD" -e MODEL_PATH="$MODEL_PATH" \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] outputs=$(ls "$WORK/repo/official-eval/outputs" 2>/dev/null | wc -l) DONE $(date +%T)"
