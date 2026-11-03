#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:25:00
#SBATCH -J hrimg12
#SBATCH -o /public/home/scnethr2601a10/probe/img12-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img12-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }

echo "== load =="
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? elapsed=$(( $(date +%s) - START ))s"

OUT=/public/home/scnethr2601a10/probe/dump-${SLURM_JOB_ID}
mkdir -p "$OUT"

echo "== dump files =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v "$OUT":/dump "$IMG" bash -lc '
    cd /root
    cp -r hackathon-mimo hackathon-mimo-code.tar.gz sglang_server.log /dump/ 2>/dev/null
    tar -tzvf hackathon-mimo-code.tar.gz 2>/dev/null | head -30
    echo "=== env.sh ==="; cat hackathon-mimo/env.sh
    echo "=== run.sh ==="; cat hackathon-mimo/run.sh
    echo "=== smoke.sh ==="; cat hackathon-mimo/smoke.sh
    echo "=== env_check.sh ==="; cat hackathon-mimo/env_check.sh
    echo "=== README.md ==="; cat hackathon-mimo/README.md
    echo "=== env.log ==="; cat hackathon-mimo/env.log
    echo "=== smoke.log ==="; cat hackathon-mimo/smoke.log
    echo "=== test_acc.py ==="; cat hackathon-mimo/test_acc.py
    echo "=== baselines ==="; ls -la hackathon-mimo/baselines; head -c 800 hackathon-mimo/baselines/* 2>/dev/null
    echo "=== data ==="; ls -la hackathon-mimo/data
    echo "=== bench ==="; ls -la hackathon-mimo/bench
    echo "=== patches ==="; ls -la hackathon-mimo/patches
    echo "=== .bash_history (last 60) ==="; tail -60 /root/.bash_history
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== device visibility experiments =="
cat > /tmp/dev.sh <<'EOS'
for MODE in "a" "b" "c"; do
  case $MODE in
    a) FLAGS="--device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined" ;;
    b) FLAGS="--privileged" ;;
    c) FLAGS="--device /dev/kfd --device /dev/dri --group-add video --userns=keep-id --security-opt seccomp=unconfined" ;;
  esac
  echo "### mode=$MODE flags=$FLAGS"
done
EOS
for MODE in 1 2 3; do
  case $MODE in
    1) FLAGS="--device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined" ;;
    2) FLAGS="--privileged" ;;
    3) FLAGS="--device /dev/kfd --device /dev/dri --group-add video --security-opt seccomp=unconfined --userns=keep-id" ;;
  esac
  echo "### mode=$MODE  ($FLAGS)"
  $PD run --rm --ipc=host $FLAGS "$IMG" bash -lc '
    ls -l /dev/kfd 2>&1 | head -1
    export LD_LIBRARY_PATH=/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
    python3 -c "
import torch
print(\"  available:\", torch.cuda.is_available(), \"count:\", torch.cuda.device_count())
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    print(\"  dev0:\", p.name, round(p.total_memory/2**30,1), getattr(p,\"gcnArchName\",\"?\"))
" 2>&1 | tail -4
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning" | head -6
done

echo "== DONE $(date +%T) =="
