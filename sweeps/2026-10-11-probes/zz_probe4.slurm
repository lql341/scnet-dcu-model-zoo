#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 00:12:00
#SBATCH -J hrprobe4
#SBATCH --gres=dcu:1

module purge
module load compiler/gcc/9.3.0 compiler/dtk/26.04 2>/dev/null

echo "===== node ====="
hostname; nproc
echo "DTKROOT=$DTKROOT  ROCM_PATH=$ROCM_PATH"

echo "===== hy-smi ====="
command -v hy-smi || ls /opt/hyhal/bin/hy-smi 2>/dev/null
hy-smi 2>&1 | head -20

echo "===== hy-smi --showproductname ====="
hy-smi -a --json 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    for c in d.get('card',[]):
        print({k:c.get(k) for k in ('Card series','Card model','Card SKU','Card vendor','Perf Level','FB Usage')})
except Exception as e: print('json parse fail',e)
" 2>/dev/null || true

echo "===== torch ====="
command -v python3 && python3 -V
python3 - <<'PY' 2>&1 | tail -25
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("is_available:", torch.cuda.is_available(), "count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} vram={p.total_memory/2**30:.1f}GB cu={getattr(p,'multi_processor_count','?')} arch={getattr(p,'gcnArchName','?')}")
    if torch.cuda.is_available():
        import time
        x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); t=time.time()
        for _ in range(20): y = x @ x
        torch.cuda.synchronize(); dt=time.time()-t
        print(f"bf16 4096^3 x20: {dt/20*1000:.1f} ms/iter -> {2*4096**3*20/dt/1e12:.1f} TFLOPS")
        xf = torch.randn(1024,1024,device="cuda",dtype=torch.float8_e4m3fn) if hasattr(torch,'float8_e4m3fn') else None
        print("float8 dtype present:", xf is not None)
except Exception:
    traceback.print_exc()
PY

echo "===== packages ====="
python3 -m pip list 2>/dev/null | grep -iE "^(torch|triton|vllm|sglang|transformers|flash)" | head

echo "===== inter-node net ====="
ip -o -4 addr show 2>/dev/null | awk '{print $2,$4}' | head -6

echo "===== DONE ====="
