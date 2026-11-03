#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J hrimg9
#SBATCH -o /public/home/scnethr2601a10/probe/img9-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img9-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

mkdir -p /tmp/chk
cat > /tmp/chk/check.py <<'PY'
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print("  [%d] %s vram=%.1fGB arch=%s cu=%s" % (i, p.name, p.total_memory/2**30, getattr(p,"gcnArchName","?"), getattr(p,"multi_processor_count","?")))
    if torch.cuda.is_available():
        import time
        x = torch.randn(4096,4096,device="cuda",dtype=torch.bfloat16)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(30): y = x @ x
        torch.cuda.synchronize(); dt=time.time()-t
        print("bf16 4096^3: %.1f ms/iter -> %.1f TFLOPS" % (dt/30*1000, 2*4096**3*30/dt/1e12))
except Exception:
    traceback.print_exc()
PY

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
PUB=$(df -Pm /public | awk 'NR==2{print int($4/1024)}'); echo "public free=${PUB}G"

# 方案1：graphroot 放 /public/home（NFS，跨作业持久，image 只 load 一次）
PD_NFS="podman --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"
# 方案2：graphroot 放 /tmp（节点本地）
PD_TMP="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== attempt NFS graphroot (default) =="
START=$(date +%s)
$PD_NFS load -i "$TAR" >/tmp/load1.log 2>&1; RC=$?
echo "nfs_load_rc=$RC elapsed=$(( $(date +%s) - START ))s"
grep -iE "^Error|invalid|not supported|permission" /tmp/load1.log | tail -3

PD="$PD_NFS"
if [ $RC -ne 0 ] || ! $PD_NFS image exists "$IMG" >/dev/null 2>&1; then
  echo "== fallback: /tmp graphroot =="
  if [ "$FREE" -lt 45 ]; then echo "NOT_ENOUGH_SPACE_ON_THIS_NODE tmp=${FREE}G ; exit 42"; exit 42; fi
  START=$(date +%s)
  $PD_TMP load -i "$TAR" >/tmp/load2.log 2>&1; RC=$?
  echo "tmp_load_rc=$RC elapsed=$(( $(date +%s) - START ))s"
  grep -iE "^Error|invalid" /tmp/load2.log | tail -3
  PD="$PD_TMP"
fi

echo "== images =="
$PD images --format '{{.Repository}}:{{.Tag}} {{.Size}}' 2>/dev/null | head -3

echo "== run check =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v /tmp/chk:/chk:ro "$IMG" bash -lc '
    python3 /chk/check.py
    echo "-- versions --"
    for m in sglang vllm transformers triton; do python3 -c "import $m;print(\"$m:\",getattr($m,\"__version__\",\"?\"))" 2>/dev/null; done
    echo "-- sglang version --"; sglang version 2>&1 | head -4
    echo "-- serve subcmd --"; sglang serve --help 2>&1 | head -5
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
