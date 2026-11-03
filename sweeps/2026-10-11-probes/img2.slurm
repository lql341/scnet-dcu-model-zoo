#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:25:00
#SBATCH -J hrimg2
#SBATCH -o /public/home/scnethr2601a10/probe/img2-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img2-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p /tmp/emptycerts
PODMAN="podman --root /tmp/pr --runroot /tmp/prr"

echo "===== 0. node ====="; hostname
echo "===== 1. certs.d diagnostic ====="
ls -la /etc/docker/certs.d/ 2>&1 | head
ls -la "/etc/docker/certs.d/image.ac.com:5000/" 2>&1 | head
for f in /etc/docker/certs.d/image.ac.com:5000/*; do printf "%-70s readable=" "$f"; [ -r "$f" ] && echo YES || echo NO; done 2>/dev/null

echo "===== 2. try pull variants ====="
try() { echo "--- variant: $* ---"; timeout 600 $PODMAN pull "$@" "$IMG" 2>&1 | grep -viE "subuid|rootless single|Network file system|level=warning" | tail -3; }

try --tls-verify=false --cert-dir=/tmp/emptycerts
if ! $PODMAN image exists "$IMG" 2>/dev/null; then
  CA=$(ls /etc/docker/certs.d/image.ac.com:5000/*.crt 2>/dev/null | head -1)
  if [ -n "$CA" ] && [ -r "$CA" ]; then
    mkdir -p /tmp/mycerts/image.ac.com:5000 && cp "$CA" /tmp/mycerts/image.ac.com:5000/ca.crt
    echo "copied CA: $CA"
    try --tls-verify=false --cert-dir=/tmp/mycerts
  fi
fi
if ! $PODMAN image exists "$IMG" 2>/dev/null; then
  echo "--- variant: --tls-verify=false with CONTAINERS_CERTS ---"
  try --tls-verify=false
fi

echo "===== 3. result ====="
$PODMAN image exists "$IMG" && echo "IMAGE_PRESENT=YES" || echo "IMAGE_PRESENT=NO"
$PODMAN images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -viE "level=" | head -3

echo "===== 4. run validation ====="
if $PODMAN image exists "$IMG"; then
$PODMAN run --rm --ipc=host --shm-size=16g \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  "$IMG" bash -lc '
    python3 - <<PY 2>&1 | tail -18
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p=torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} vram={p.total_memory/2**30:.1f}GB arch={getattr(p,\"gcnArchName\",\"?\")}")
    if torch.cuda.is_available():
        import time; x=torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(20): y=x@x
        torch.cuda.synchronize(); dt=time.time()-t
        print(f"bf16 4096^3 {dt/20*1000:.1f} ms -> {2*4096**3*20/dt/1e12:.1f} TFLOPS")
except Exception: traceback.print_exc()
PY
    echo "-- versions --"
    for m in sglang vllm transformers triton fla; do python3 -c "import $m;print(\"$m:\",getattr($m,\"__version__\",\"?\"))" 2>/dev/null; done
    echo "-- sglang cli --"; sglang --version 2>&1 | head -2
    echo "-- model dirs --"; ls -d /models /workspace /data /root/models 2>/dev/null
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"
fi

echo "===== DONE $(date +%T) ====="
