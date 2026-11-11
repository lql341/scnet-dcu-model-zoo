#!/bin/bash -l
# 一键复现（BW1000 + 官方 Wan2.1 代码）：1 卡 1 进程 × N 卡并行跑官方 60 条条件
#   sbatch container/submit.slurm            # 用 NODES/GRES 默认 8 卡
#   PROBE_ONLY=1 sbatch container/submit.slurm   # 只跑第 0 片的前 1 条（探针）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4          # 本集群 sbatch 命令行 --gres 覆盖无效，只能在文件头改
#SBATCH -c 32
#SBATCH -t 2-00:00:00
#SBATCH -J wan-i2v
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
WORK="${WORK:-/public/home/scnethr2601a10/work/zoo-wan}"
# 官方素材（首帧图）在 official-t34 下，只读挂进 /assets
ASSETS="${ASSETS:-/public/home/scnethr2601a10/work/official-t34}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G"
# 只 load 一个镜像（解压约 32 GB），45 GB 足以；阈值设太高会连续被节点水位挡掉（实测三次 65–69G 被拒）
[ "$FREE" -lt 45 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

if [ -n "${HOST_IMAGE_TAR:-}" ] && [ -s "$HOST_IMAGE_TAR" ]; then $PD load -i "$HOST_IMAGE_TAR" >/tmp/load-$JID.log 2>&1; fi
$PD image exists "$IMAGE" 2>/dev/null || $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMAGE" >/tmp/load-$JID.log 2>&1
$PD image exists "$IMAGE" || { echo "[host] image unavailable"; tail -3 /tmp/load-$JID.log; exit 1; }

mkdir -p "$WORK/repo" && cp -rf "$REPO"/. "$WORK/repo"/ 2>/dev/null
# 卡数：本集群不设 HIP_VISIBLE_DEVICES（实测 dcu:4 的作业里该变量为空/0），
# 用 scontrol 查 AllocTRES 里的 gres/dcu 最可靠；退回 CUDA_VISIBLE_DEVICES 个数
NCARD=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -m1 -oE 'gres/dcu=[0-9]+' | cut -d= -f2)
[ -z "${NCARD:-}" ] && NCARD=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')
[ "${NCARD:-0}" -lt 1 ] && NCARD=1
# 可见性：本集群 Slurm 不设 HIP_VISIBLE_DEVICES，容器默认能看到 cgroup 允许的全部卡。
# 【坑】若传 -e HIP_VISIBLE_DEVICES=""（空串）会被 torch 解释成"零张卡"，
# 报 `RuntimeError: No HIP GPUs are available`（实测踩过）。所以空值就不要传。
VIS=""   # 一律不传：容器靠 cgroup 限制，torchrun 自己枚举
VISARGS=()
[ -n "$VIS" ] && VISARGS=(-e HIP_VISIBLE_DEVICES="$VIS" -e CUDA_VISIBLE_DEVICES="$VIS")
echo "[host] 卡数=$NCARD (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset})  run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  "${VISARGS[@]}" \
  -v /opt/hyhal:/opt/hyhal:ro \
  -v "$HOST_MODEL_PATH":$MODEL_PATH:ro -v "$WORK":/work -v "$ASSETS":/assets:ro \
  -v "$WAN_SRC_HOST":/wan-src:ro -v "$WHEELS_HOST":/wheels:ro -v "$WHEELS_QI_HOST":/wheels-qi:ro \
  -v "$REPO/container/inner.sh":/inner.sh:ro \
  -e MODEL_PATH="$MODEL_PATH" -e WAN_SRC=/wan-src -e SHARD_TOTAL="$NCARD" -e PROBE_ONLY="${PROBE_ONLY:-0}" -e OFFLOAD_MODEL="${OFFLOAD_MODEL:-True}" -e ASSET_ROOT=/assets -e T5_CPU="${T5_CPU:-1}" -e FSDP="${FSDP:-0}" -e FSDP_NPROC="$NCARD" \
  "$IMAGE" /inner.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning"
echo "[host] outputs=$(ls "$WORK/repo/outputs" 2>/dev/null | wc -l)  DONE $(date +%T)"
