#!/bin/bash -l
# 归档 Music3 镜像（防上游删除）：policy 见 docs/image-archive.md
# 压缩前 12.27 GB，`podman save | gzip -1` 后约 12 GB；需要节点 /tmp >= 55 G
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 02:00:00
#SBATCH -J arch-m3
#SBATCH -o /public/home/scnethr2601a10/work/arch-m3-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/arch-m3-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1
OUT=/public/home/scnethr2601a10/zoo-images/minimax-music3-bw-sglang_20261031-rc1.tar.gz
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 60 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
[ -s "$OUT" ] && { echo "[host] 已存在，跳过"; ls -la "$OUT"; exit 0; }
$PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1 || { tail -3 /tmp/pull.log; exit 1; }
echo "[host] 镜像就绪，开始 save|gzip $(date +%T)"
$PD save "$IMG" | gzip -1 > "$OUT" || { echo "[host] save 失败"; rm -f "$OUT"; exit 1; }
echo "[host] 完成 $(date +%T)"; ls -la "$OUT"; sha256sum "$OUT"
