#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -J hrimg5
#SBATCH -o /public/home/scnethr2601a10/probe/img5-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img5-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/dev/shm/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p /dev/shm/emptycerts

# 节点根盘经常 99% 满，改用 /dev/shm（tmpfs，504G）；rootless 无 subuid 必须 ignore_chown_errors
GROOT=/dev/shm/podman-root
PD="podman --root $GROOT --runroot /dev/shm/podman-run --storage-driver vfs --storage-opt ignore_chown_errors=true"
LOG=/public/home/scnethr2601a10/probe/pull-${SLURM_JOB_ID}.log

echo "== node $(hostname) $(date +%T) =="
echo "== disk =="; df -h / /tmp /dev/shm 2>/dev/null | grep -vE "^Filesystem" | head -4

echo "== PULL =="
START=$(date +%s)
$PD pull --tls-verify=false --cert-dir=/dev/shm/emptycerts "$IMG" > "$LOG" 2>&1
RC=$?
echo "pull_rc=$RC elapsed=$(( $(date +%s) - START ))s"
echo "== disk after =="; df -h /dev/shm | tail -1
if [ $RC -ne 0 ]; then
  echo "== ERRORS =="; grep -iE "^Error|denied|space|EOF|unauthorized|chown|invalid" "$LOG" | tail -8
  echo "== tail =="; tail -5 "$LOG"
  exit 1
fi

echo "== images =="; $PD images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -3

echo "== validate =="
$PD run --rm --ipc=host --shm-size=16g \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined "$IMG" bash -lc '
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
    for m in sglang vllm transformers triton; do python3 -c "import $m;print(\"$m:\",getattr($m,\"__version__\",\"?\"))" 2>/dev/null; done
    echo "-- sglang cli --"; sglang --version 2>&1 | head -2
    echo "-- dirs --"; ls -d /models /workspace /data /root/models 2>/dev/null
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== save image to /public for reuse =="
SAVE=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
mkdir -p /public/home/scnethr2601a10/images
if [ ! -s "$SAVE" ]; then
  time $PD save -o "$SAVE" "$IMG" 2>/dev/null
fi
ls -lh "$SAVE" 2>/dev/null

echo "== DONE $(date +%T) =="
