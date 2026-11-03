#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J hrimg7
#SBATCH -o /public/home/scnethr2601a10/probe/img7-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img7-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR" /tmp/emptycerts
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free = ${FREE} GB"
if [ "$FREE" -lt 45 ]; then echo "NOT_ENOUGH_SPACE_ON_THIS_NODE (${FREE}G < 45G), exit 42"; exit 42; fi

echo "== LOAD from tar =="
START=$(date +%s)
$PD load -i "$TAR" >/tmp/load.log 2>&1
RC=$?
echo "load_rc=$RC elapsed=$(( $(date +%s) - START ))s"
if [ $RC -ne 0 ]; then
  echo "-- load err --"; grep -iE "^Error|invalid|EOF" /tmp/load.log | tail -3
  echo "== FALLBACK pull =="
  START=$(date +%s)
  $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/load.log 2>&1
  RC=$?; echo "pull_rc=$RC elapsed=$(( $(date +%s) - START ))s"
  [ $RC -ne 0 ] && { grep -iE "^Error" /tmp/load.log | tail -3; exit 1; }
fi

echo "== image =="
$PD images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -3
echo "-- env --"
$PD image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | grep -viE "level=" | grep -iE "PATH|DTK|ROCM|PYTHON|LD_LIB|SGLANG|HF_|MODEL" | head -12
echo "-- entrypoint/cmd/workdir --"
$PD image inspect "$IMG" --format 'EP={{json .Config.Entrypoint}} CMD={{json .Config.Cmd}} WD={{.Config.WorkingDir}}' 2>/dev/null | grep -viE "level="

echo "== VALIDATE (--ipc=host, no --shm-size) =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined "$IMG" bash -lc '
    python3 - <<PY 2>&1 | tail -20
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
    echo "-- dtk --"; ls -d /opt/dtk* 2>/dev/null | head -2
    echo "-- dsmi --"; command -v hy-smi dsmi 2>/dev/null
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
