#!/bin/bash -l
# 验证归档可还原：load 后 digest 必须与 MANIFEST 记录一致
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:50:00
#SBATCH -J img-verify
#SBATCH -o /public/home/scnethr2601a10/zoo-images/verify-%j.out
#SBATCH -e /public/home/scnethr2601a10/zoo-images/verify-%j.err
set -uo pipefail
D=/public/home/scnethr2601a10/zoo-images
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 80 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

ONLY=${ONLY:-all}
for f in "$D"/mimo-sglang-bw10000_1.1.1.tar.gz "$D"/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz; do
  case "$ONLY" in sgl) case "$f" in *vllm*) continue;; esac;; vllm) case "$f" in *sglang*) continue;; esac;; esac
  echo "== load $(basename $f) $(date +%T) =="
  gunzip -c "$f" | $PD load
done
echo "== 本地镜像与 digest =="
$PD images --format '{{.Repository}}:{{.Tag}}|{{.Digest}}|{{.Size}}' 2>/dev/null
echo "== DONE $(date +%T) =="
