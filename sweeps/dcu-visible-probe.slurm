#!/bin/bash -l
# 诊断：Ci 申请 4 张卡时，容器里 torch 能看到几张？（OmniWeaving 序列并行 build_mesh 失败的原因）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -J vis-probe
#SBATCH -o /public/home/scnethr2601a10/work/vis-probe-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp_free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
echo "== host env: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset} renderD=$(ls /dev/dri/renderD* 2>/dev/null | wc -l)"
scancel_dummy=1
for MODE in "plain" "hvd04" "cvd04"; do
  case $MODE in
    plain) ARGS=() ;;
    hvd04) ARGS=(-e HIP_VISIBLE_DEVICES=0,1,2,3) ;;
    cvd04) ARGS=(-e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3) ;;
  esac
  echo "-- mode=$MODE ${ARGS[*]:-}"
  $PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
    -e LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/usr/local/lib \
    "${ARGS[@]}" "$IMG" -c 'python3 -c "
import torch, os
print(\"  device_count=\", torch.cuda.device_count(), \"| HIP_VISIBLE_DEVICES=\", os.environ.get(\"HIP_VISIBLE_DEVICES\"), \"| CUDA_VISIBLE_DEVICES=\", os.environ.get(\"CUDA_VISIBLE_DEVICES\"))
"' 2>&1 | grep -viE "subuid|rootless|level=warning" | tail -2
done
