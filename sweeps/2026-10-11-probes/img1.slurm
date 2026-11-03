#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:25:00
#SBATCH -J hrimg1
#SBATCH -o /public/home/scnethr2601a10/probe/img-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img-%j.err

set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PROOT=/tmp/podman-root
mkdir -p $PROOT
PODMAN="podman --root $PROOT --runroot /tmp/podman-run --storage-driver overlay"

echo "===== 0. node ====="; hostname; date
hy-smi 2>/dev/null | grep -c HCU | sed 's/^/HCU visible: /' || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}  HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"

echo "===== 1. pull ($(date +%T)) ====="
time $PODMAN pull --tls-verify=false "$IMG" 2>&1 | grep -viE "subuid|rootless single|Network file system|level=warning" | tail -8
echo "pull exit=$?"

echo "===== 2. image metadata ====="
$PODMAN image inspect "$IMG" --format '{{json .Config.Env}}' 2>&1 | grep -viE "level=" | head -3
echo "--- Entrypoint/Cmd ---"
$PODMAN image inspect "$IMG" --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}} {{json .Config.WorkingDir}}' 2>&1 | grep -viE "level=" | head -2
echo "--- size ---"
$PODMAN images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>&1 | grep -viE "level=" | head -3

echo "===== 3. run: torch / sglang / device ====="
$PODMAN run --rm --ipc=host --shm-size=16g \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  "$IMG" bash -lc '
    echo "-- whoami --"; whoami
    echo "-- python --"; python3 -V 2>&1 || python -V 2>&1
    echo "-- torch --"
    python3 - <<PY 2>&1 | tail -20
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} vram={p.total_memory/2**30:.1f}GB arch={getattr(p,\"gcnArchName\",\"?\")}")
    if torch.cuda.is_available():
        import time
        x = torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(20): y = x @ x
        torch.cuda.synchronize(); dt=time.time()-t
        print(f"bf16 4096^3: {dt/20*1000:.1f} ms/iter -> {2*4096**3*20/dt/1e12:.1f} TFLOPS")
except Exception:
    traceback.print_exc()
PY
    echo "-- sglang --"
    python3 -c "import sglang;print(\"sglang:\", sglang.__version__)" 2>&1 | tail -2
    python3 -c "import vllm;print(\"vllm:\", vllm.__version__)" 2>&1 | tail -2
    python3 -c "import transformers;print(\"transformers:\", transformers.__version__)" 2>&1 | tail -2
    echo "-- dtk --"; ls -d /opt/dtk* 2>/dev/null | head -3; cat /opt/dtk/.info/version 2>/dev/null | head -2
    echo "-- sglang cli --"; (sglang --version 2>&1 || python3 -m sglang.launch_server --help >/dev/null 2>&1 && echo "launch_server OK") | head -3
    echo "-- model dirs --"; ls -d /models /workspace /root/* /data 2>/dev/null | head -10
    echo "-- pip key pkgs --"; pip list 2>/dev/null | grep -iE "^(torch|triton|sglang|vllm|transformers|fla)" | head -10
  ' 2>&1 | grep -viE "subuid|rootless single|Network file system|level=warning|level=error"

echo "===== 4. cleanup ====="
$PODMAN rmi -f "$IMG" >/dev/null 2>&1; rm -rf $PROOT /tmp/podman-run "$XDG_RUNTIME_DIR"
echo "===== DONE $(date +%T) ====="
