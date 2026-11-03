#!/bin/bash -l
# 补：vLLM 镜像归档（podman 4.9 的 save --compress 只支持 docker-dir，改用 save | gzip）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:50:00
#SBATCH -J img-arch-vllm
#SBATCH -o /public/home/scnethr2601a10/zoo-images/archv-%j.out
#SBATCH -e /public/home/scnethr2601a10/zoo-images/archv-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
DEST=/public/home/scnethr2601a10/zoo-images
mkdir -p "$DEST"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
OUT="$DEST/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 70 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

echo "== pull $(date +%T) =="
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1 \
  || { echo PULL_FAIL; tail -3 /tmp/pull.log; exit 1; }

echo "== save|gzip $(date +%T) =="
$PD save "$IMG" | gzip -1 > "$OUT.tmp"
mv "$OUT.tmp" "$OUT"
ls -lh "$OUT"

echo "== 校验（解压后 load 的 manifest 应与原 digest 一致）=="
sha256sum "$OUT"
chmod 755 "$DEST"; chmod 644 "$DEST"/* 2>/dev/null
ls -la "$DEST"
echo "== DONE $(date +%T) =="
