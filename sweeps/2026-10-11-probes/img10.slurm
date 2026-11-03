#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J hrimg10
#SBATCH -o /public/home/scnethr2601a10/probe/img10-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img10-%j.err

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

cat > /tmp/chk.sh <<'EOS'
# 找出 rocm_smi 并用完整 LD_LIBRARY_PATH
set -u
echo "-- locate librocm_smi64 --"
find /opt/dtk /usr/lib /usr/local/lib -name 'librocm_smi64*' 2>/dev/null | head -5
export LD_LIBRARY_PATH=/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
echo "-- python version --"; python3 -V
echo "-- torch --"
python3 -c "
import torch
print('torch:', torch.__version__, 'hip:', torch.version.hip)
print('available:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print('  [%d] %s vram=%.1fGB arch=%s cu=%s' % (i,p.name,p.total_memory/2**30,getattr(p,'gcnArchName','?'),getattr(p,'multi_processor_count','?')))
if torch.cuda.is_available():
    import time
    x=torch.randn(4096,4096,device='cuda',dtype=torch.bfloat16)
    torch.cuda.synchronize(); t=time.time()
    for _ in range(30): y=x@x
    torch.cuda.synchronize(); dt=time.time()-t
    print('bf16 4096^3: %.1f ms/iter -> %.1f TFLOPS' % (dt/30*1000, 2*4096**3*30/dt/1e12))
" 2>&1 | tail -12
echo "-- sglang serve --"; export PYTHONPATH=; sglang serve --help 2>&1 | head -4
echo "-- model dirs --"
for d in /models /workspace /data /root /opt/models /mnt; do [ -d "$d" ] && { echo "[$d]"; ls "$d" 2>/dev/null | head -6; }; done
echo "-- find config.json (model weights) --"
timeout 40 find / -maxdepth 5 -name 'config.json' -not -path '*/site-packages/*' -not -path '/proc/*' 2>/dev/null | head -8
EOS

echo "== run =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v /tmp/chk.sh:/chk.sh:ro "$IMG" bash /chk.sh 2>&1 \
  | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
