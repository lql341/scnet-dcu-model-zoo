#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -J hrimg6
#SBATCH -o /public/home/scnethr2601a10/probe/img6-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img6-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
export XDG_RUNTIME_DIR=/dev/shm/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
mkdir -p /dev/shm/emptycerts
LOG=/public/home/scnethr2601a10/probe/pull-${SLURM_JOB_ID}.log

# rootless + 无 /etc/subuid → 必须 fuse-overlayfs + ignore_chown_errors
# 节点 / 盘经常 100% 满 → graphroot 放 /dev/shm (tmpfs 504G)
PD="podman --root /dev/shm/pr --runroot /dev/shm/prr \
    --storage-driver fuse-overlayfs --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
df -h /dev/shm | tail -1

echo "== PULL =="
START=$(date +%s)
$PD pull --tls-verify=false --cert-dir=/dev/shm/emptycerts "$IMG" > "$LOG" 2>&1
RC=$?
echo "pull_rc=$RC elapsed=$(( $(date +%s) - START ))s"
df -h /dev/shm | tail -1

if [ $RC -ne 0 ]; then
  echo "== ERRORS =="; grep -iE "^Error|denied|space|EOF|unauthorized|chown|invalid|not supported" "$LOG" | tail -8
  echo "== tail =="; tail -4 "$LOG"
  echo "== FALLBACK: overlay on /tmp =="
  FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
  echo "tmp free = ${FREE} GB"
  PD2="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"
  if [ "$FREE" -ge 20 ]; then
    START=$(date +%s)
    $PD2 pull --tls-verify=false --cert-dir=/dev/shm/emptycerts "$IMG" >> "$LOG" 2>&1
    RC=$?
    echo "fallback_rc=$RC elapsed=$(( $(date +%s) - START ))s"
    [ $RC -ne 0 ] && { grep -iE "^Error" "$LOG" | tail -3; exit 1; }
    PD="$PD2"
  else
    echo "no space on /tmp, abort"; exit 1
  fi
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
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== save image tar =="
SAVE=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
mkdir -p /public/home/scnethr2601a10/images
[ -s "$SAVE" ] || $PD save -o "$SAVE" "$IMG" >/dev/null 2>&1
ls -lh "$SAVE" 2>/dev/null

echo "== DONE $(date +%T) =="
