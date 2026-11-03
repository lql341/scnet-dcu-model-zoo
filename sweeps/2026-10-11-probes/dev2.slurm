#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -J hrdev2
#SBATCH -o /public/home/scnethr2601a10/probe/dev2-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/dev2-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
echo "=== HOST /dev (dcu related) ==="
ls -l /dev/kfd /dev/dri/ 2>&1 | head -14
ls -l /dev | grep -iE "hydmi|hsa|kfd|dri|dcu" | head
echo "=== HOST /sys/class/kfd ==="
ls /sys/class/kfd/kfd/topology/nodes/ 2>/dev/null | head
echo "=== HOST hy-smi ==="
hy-smi 2>&1 | grep -E "^[0-9]" | head -3
echo "=== HOST cgroup device ==="
cat /proc/self/cgroup | head -3

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }
echo "== load =="
$PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$?"

cat > /tmp/dev.sh <<'EOS'
LP=/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/.hyhal/NanoLog/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/hyhal/lib:/opt/hyhal/lib64:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib
export LD_LIBRARY_PATH=$LP:${LD_LIBRARY_PATH:-}
echo "-- container /dev --"; ls -l /dev/kfd /dev/dri 2>&1 | head -12
echo "-- container /sys/class/kfd --"; ls /sys/class/kfd/kfd/topology/nodes/ 2>&1 | head -5
echo "-- container hy-smi --"; (hy-smi 2>&1 || /opt/hyhal/bin/hy-smi 2>&1) | grep -E "^[0-9]|Error|error" | head -6
echo "-- hsa debug --"; HSA_ENABLE_DEBUG=1 python3 -c "import torch;print('available',torch.cuda.is_available())" 2>&1 | grep -iE "kfd|node|error|fail|available" | head -12
EOS

echo "### CFG-A: privileged + /sys bind"
$PD run --rm --ipc=host --privileged -v /sys:/sys:ro -v /tmp/dev.sh:/dev.sh:ro "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | head -40

echo "### CFG-B: devices + /sys bind + keep-groups"
$PD run --rm --ipc=host --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined -v /sys:/sys:ro -v /tmp/dev.sh:/dev.sh:ro "$IMG" bash /dev.sh 2>&1 | grep -viE "subuid|rootless|Network file|level=warning" | head -40

echo "== DONE $(date +%T) =="
