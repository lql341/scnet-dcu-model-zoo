#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -J hrdev3
#SBATCH -o /public/home/scnethr2601a10/probe/dev3-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/dev3-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
DTKHOST=/public/software/compiler/dtk-26.04
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "node=$(hostname) tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$?"

cat > /tmp/dev.sh <<'EOS'
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
echo "-- /dev entries --"; ls /dev | grep -iE "kfd|dri|hydmi" ; ls /dev/dri | tr '\n' ' '; echo
echo "-- hy-smi presence --"; ls /opt/hyhal/bin 2>/dev/null | head -5
echo "-- hy-smi run --"; /opt/hyhal/bin/hy-smi 2>&1 | head -8
echo "-- hsa full debug --"
HSA_ENABLE_DEBUG=1 python3 -c "
import torch
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
" 2>&1 | tail -25
EOS

for CFG in 1 2 3; do
  case $CFG in
    1) F="--device /dev/kfd --device /dev/mkfd --device /dev/dri --group-add keep-groups --group-add render" ;;
    2) F="--device /dev/kfd --device /dev/mkfd --device /dev/dri --privileged -v /opt/hyhal:/opt/hyhal:ro -v $DTKHOST:/opt/dtk:ro" ;;
    3) F="--device /dev/kfd --device /dev/mkfd --device /dev/dri --group-add keep-groups --pid=host -v /sys:/sys:ro" ;;
  esac
  echo "########## CFG$CFG : $F"
  $PD run --rm --ipc=host $F --security-opt seccomp=unconfined -v /tmp/dev.sh:/dev.sh:ro "$IMG" bash /dev.sh 2>&1 \
    | grep -viE "subuid|rootless|Network file|level=warning" | head -45
done
echo "== DONE $(date +%T) =="
