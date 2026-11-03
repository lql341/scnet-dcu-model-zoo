#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:30:00
#SBATCH -J hrimg8
#SBATCH -o /public/home/scnethr2601a10/probe/img8-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/img8-%j.err

IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
PD="podman --root /tmp/pr --runroot /tmp/prr --storage-driver overlay --storage-opt ignore_chown_errors=true"

mkdir -p /tmp/chk
cat > /tmp/chk/check.py <<'PY'
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        arch = getattr(p, "gcnArchName", "?")
        print("  [%d] %s vram=%.1fGB arch=%s cu=%s" % (i, p.name, p.total_memory/2**30, arch, getattr(p,"multi_processor_count","?")))
    if torch.cuda.is_available():
        import time
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(30):
            y = x @ x
        torch.cuda.synchronize(); dt = time.time() - t
        print("bf16 4096^3: %.1f ms/iter -> %.1f TFLOPS" % (dt/30*1000, 2*4096**3*30/dt/1e12))
        # fp8 support probe
        if hasattr(torch, "float8_e4m3fn"):
            try:
                a = torch.randn(1024,1024,device="cuda").to(torch.float8_e4m3fn)
                b = torch.randn(1024,1024,device="cuda").to(torch.float8_e4m3fn)
                _ = torch._scaled_mm(a, b.t(), out_dtype=torch.bfloat16, scale_a=torch.tensor(1.0,device="cuda"), scale_b=torch.tensor(1.0,device="cuda"))
                print("fp8 _scaled_mm: OK")
            except Exception as e:
                print("fp8 _scaled_mm: FAIL", type(e).__name__, str(e)[:80])
except Exception:
    traceback.print_exc()
PY

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 45 ] && { echo "NOT_ENOUGH_SPACE, exit 42"; exit 42; }

echo "== load =="
START=$(date +%s); $PD load -i "$TAR" >/tmp/load.log 2>&1; echo "load_rc=$? elapsed=$(( $(date +%s) - START ))s"

echo "== run check =="
$PD run --rm --ipc=host \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined \
  -v /tmp/chk:/chk:ro "$IMG" bash -lc '
    python3 /chk/check.py
    echo "-- versions --"
    for m in sglang vllm transformers triton flashinfer; do python3 -c "import $m;print(\"$m:\",getattr($m,\"__version__\",\"?\"))" 2>/dev/null; done
    echo "-- sglang version --"; sglang version 2>&1 | head -3
    echo "-- serve help --"; sglang serve --help 2>&1 | head -6
    echo "-- pip das pkgs --"; pip list 2>/dev/null | grep -iE "das|sglang|torch|triton|fla" | head -10
  ' 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"

echo "== DONE $(date +%T) =="
