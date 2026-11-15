#!/bin/bash -l
# 8 卡实验（BW1000 + 官方 Wan2.1 代码）：默认走**数据并行分片**（1 卡 1 条，8 条并发）；
#   设 FSDP=1 则改走 8 卡 FSDP（一条视频由 8 卡共同推理，逐条串行）。
#   理由：官方 60 条是用 4 卡 FSDP 跑的，18.6 min/条、总计 17.8 h；而 FSDP 是"用 4 张卡降低单条延迟"，
#   分片是"用 4 张卡提高吞吐"，后者才是 60 条这种批量任务的正解（约 12.5 h）。8 卡分片预期 ~6.3 h。
#   与 submit.slurm 的差别只在配额：本集群按 socket 绑卡，每张 DCU 需 ≥8 CPU，
#   8 卡必须 -c 64；--mem 受分区 DefMemPerCPU=3888M 约束 ⇒ 64×3888M ≈ 243G，取 240G。
#   官方 60 条产物已由 4 卡分片版产出（见 README §7），本脚本用于测 8 卡 FSDP 单条耗时。
#   sbatch container/submit.slurm            # 用 NODES/GRES 默认 8 卡
#   PROBE_ONLY=1 sbatch container/submit-8card.slurm   # 只跑 1 条（计时探针）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:8          # 8 卡（命令行 --gres 覆盖无效，必须改文件头）
#SBATCH -c 64          # 8 卡 × 8 CPU（少给会被调度器静默降到 1 张，见 gotchas #30）
#SBATCH --mem=240G
#SBATCH -t 2-00:00:00
#SBATCH -J wan-i2v-8
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在条目根目录提交"; exit 1; }
source "$REPO/env.sh"
# WORK 必须与 REPO 分开：若相等，下面的 cp 会把目录拷进自己，GNU cp 直接报错退出，
# 结果 /work/repo 缺 patches（patch 没打上），运行时报 ModuleNotFoundError: dashscope。
WORK="${WORK:-/public/home/luql/work/wan8}"
# 官方素材（首帧图）在 official-t34 下，只读挂进 /assets
ASSETS="${ASSETS:-/public/home/luql/work/official-t34}"
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

mkdir -p "$WORK/repo"
if [ "$(cd "$REPO" && pwd -P)" = "$(cd "$WORK/repo" && pwd -P)" ]; then
  echo "[host] WORK/repo 与 REPO 相同，跳过拷贝（避免 cp 拷进自己）"
else
  cp -rf "$REPO"/. "$WORK/repo"/ || { echo "[host] 拷贝仓库失败"; exit 1; }
fi
# 卡数：本集群不设 HIP_VISIBLE_DEVICES（实测 dcu:4 的作业里该变量为空/0），
# 用 scontrol 查 AllocTRES 里的 gres/dcu 最可靠；退回 CUDA_VISIBLE_DEVICES 个数
NCARD=$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null | tr ' ' '\n' | grep -m1 -oE 'gres/dcu=[0-9]+' | cut -d= -f2)
[ -z "${NCARD:-}" ] && NCARD=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | awk -F, '{print NF}')
[ "${NCARD:-0}" -lt 1 ] && NCARD=1
# 可见性：本集群 Slurm 不设 HIP_VISIBLE_DEVICES，容器默认能看到 cgroup 允许的全部卡。
# 【坑】若传 -e HIP_VISIBLE_DEVICES=""（空串）会被 torch 解释成"零张卡"，
# 报 `RuntimeError: No HIP GPUs are available`（实测踩过）。所以空值就不要传。
VIS=$(seq -s, 0 $(( ${NCARD:-1} - 1 )))   # 显式写满：unset 在本集群拿不到 4 张以上（见 gotchas #30）
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
