#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 00:30:00
#SBATCH -J hrdev
#SBATCH -o /public/home/scnethr2601a10/probe/dev-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/dev-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
DTKHOST=/public/software/compiler/dtk-26.04
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE tmp=${FREE}G exit 42"; exit 42; }
echo "== load =="
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? elapsed=$(( $(date +%s) - START ))s"

cat > /tmp/dev.sh <<'EOS'
LP=/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/NanoLog/lib
LP=$LP:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib
LP=$LP:/opt/hyhal/lib:/opt/hyhal/lib64:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib:/usr/lib64:/usr/lib
export LD_LIBRARY_PATH=$LP:${LD_LIBRARY_PATH:-}
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
python3 -c "
import torch
ok = torch.cuda.is_available()
print('  RESULT available=%s count=%s' % (ok, torch.cuda.device_count()))
if ok:
    p = torch.cuda.get_device_properties(0)
    print('  dev0: %s vram=%.1fGB arch=%s' % (p.name, p.total_memory/2**30, getattr(p,'gcnArchName','?')))
" 2>&1 | grep -E "RESULT|dev0|Error|error" | tail -4
EOS

BASE="--rm --ipc=host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined -v /tmp/dev.sh:/dev.sh:ro"

echo "### CFG1: image-only + all .hyhal"
$PD run $BASE --group-add keep-groups "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -4

echo "### CFG2: image-only + --privileged"
$PD run $BASE --privileged "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -4

echo "### CFG3: +host /opt/hyhal mounted, image /opt/dtk"
$PD run $BASE --group-add keep-groups -v /opt/hyhal:/opt/hyhal:ro "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -4

echo "### CFG4: +host dtk-26.04 as /opt/dtk + host /opt/hyhal"
$PD run $BASE --group-add keep-groups -v "$DTKHOST":/opt/dtk:ro -v /opt/hyhal:/opt/hyhal:ro "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -4

echo "### CFG5: CFG4 + privileged"
$PD run $BASE --privileged -v "$DTKHOST":/opt/dtk:ro -v /opt/hyhal:/opt/hyhal:ro "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | tail -4

echo "### host /opt/hyhal contents"
ls /opt/hyhal/lib | grep -E "libhydmi|libhsa-runtime|librocm_smi" | head
echo "### host dtk .hyhal"
ls -d "$DTKHOST"/.hyhal/*/lib 2>/dev/null
echo "== DONE $(date +%T) =="
