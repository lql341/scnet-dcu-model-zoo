#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -J hrimg4
#SBATCH -o /public/home/scnethr2601a10/probe/img4-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img4-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p /tmp/emptycerts
LOG=/public/home/scnethr2601a10/probe/pull-${SLURM_JOB_ID}.log
PD="podman --root /tmp/pr --runroot /tmp/prr"

echo "== node $(hostname) $(date +%T) =="
echo "== disk before =="; df -h /tmp /dev/shm | tail -3
echo "== inode before =="; df -i /tmp | tail -1

echo "== PULL #1 (full log -> $LOG) =="
START=$(date +%s)
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" > "$LOG" 2>&1
RC=$?
echo "pull1_rc=$RC elapsed=$(( $(date +%s) - START ))s"
echo "== disk after =="; df -h /tmp | tail -1

echo "== errors in log =="
grep -iE "error|denied|space|timeout|EOF|unauthorized|not found|failed" "$LOG" | grep -viE "level=warning" | head -20
echo "== last 6 lines =="
tail -6 "$LOG"

if [ $RC -ne 0 ]; then
  echo "== PULL #2 retry =="
  START=$(date +%s)
  $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >> "$LOG" 2>&1
  RC=$?
  echo "pull2_rc=$RC elapsed=$(( $(date +%s) - START ))s"
  grep -iE "error|denied|space|timeout|EOF|unauthorized" "$LOG" | tail -5
fi

echo "== images =="
$PD images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -3
echo "== DONE $(date +%T) =="
