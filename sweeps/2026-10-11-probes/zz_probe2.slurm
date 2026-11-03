#!/bin/bash
#SBATCH -p hx1hcnormal
#SBATCH -N 1
#SBATCH -o /public/home/luql/probe/p2-%j.out
#SBATCH -e /public/home/luql/probe/p2-%j.err
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:12:00
#SBATCH -J hrprobe2

echo "===== node ====="
hostname; date; nproc

echo "===== dtk 26.04 env ====="
ls /public/software/compiler/dtk-26.04/ 2>/dev/null | head -20
for e in /public/software/compiler/dtk-26.04/env.sh /public/software/compiler/dtk-26.04/.info/version; do
  [ -e "$e" ] && { echo "--- $e"; head -5 "$e"; }
done
export LD_LIBRARY_PATH=/public/software/compiler/dtk-26.04/lib:/public/software/compiler/dtk-26.04/lib64:/public/software/compiler/dtk-26.04/hip/lib:/public/software/compiler/dtk-26.04/llvm/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH

echo "===== device nodes ====="
ls -l /dev/kfd /dev/dri 2>&1 | head -12
id
groups

echo "===== hy-smi (dtk2604 libs) ====="
hy-smi 2>&1 | head -25

echo "===== torch ====="
python3 - <<'PY' 2>&1 | head -50
import traceback
try:
    import torch
    print("torch:", torch.__version__, "hip:", torch.version.hip)
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} vram={p.total_memory/2**30:.1f}GB cu={p.multi_processor_count} arch={p.gcnArchName}")
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    import time; torch.cuda.synchronize(); t=time.time()
    for _ in range(20): y = x @ x
    torch.cuda.synchronize()
    print(f"bf16 2048^3 matmul: {(time.time()-t)/20*1000:.1f} ms  ({(2*2048**3*20)/(time.time()-t)/1e12:.1f} TFLOPS)")
except Exception:
    traceback.print_exc()
PY

echo "===== stacks ====="
python3 - <<'PY' 2>&1 | head -30
import importlib
for m in ["vllm","sglang","transformers","triton","flashinfer","vllm_plugin_das","sglang_das","fla","fla_core"]:
    try:
        mod=importlib.import_module(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e: print(f"  {m}: MISSING ({type(e).__name__}: {str(e)[:60]})")
PY

echo "===== container ====="
docker images 2>&1 | head -5
podman images 2>&1 | head -5
ls /public/ai_data/images/*.tar 2>/dev/null | head

echo "===== DONE ====="
