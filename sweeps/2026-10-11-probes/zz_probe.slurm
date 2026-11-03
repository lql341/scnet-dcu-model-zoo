#!/bin/bash
#SBATCH -p hx1hcnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:15:00
#SBATCH -J hrprobe
#SBATCH -o /public/home/luql/probe/probe-%j.out
#SBATCH -e /public/home/luql/probe/probe-%j.err

echo "===== 0. job/node ====="
date; hostname; nproc; free -g | head -2
echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"
echo "SLURM_JOB_GRES=$SLURM_JOB_GRES"
env | grep -iE "SLURM_JOB_|CUDA_VISIBLE|HIP_VISIBLE|ROCR|LD_LIBRARY_PATH" | head -20

echo "===== 1. DTK / ROCm ====="
ls -d /opt/dtk* 2>/dev/null
for f in /opt/dtk/.info/version /opt/dtk-*/share/doc/*/version /opt/hyhal/share/doc/*/version; do
  [ -e "$f" ] && { echo "--- $f"; head -5 "$f"; }
done
cat /opt/dtk/.info/* 2>/dev/null | head -20

echo "===== 2. hy-smi ====="
which hy-smi rocm-smi 2>/dev/null
hy-smi 2>&1 | head -30

echo "===== 3. topology ====="
hy-smi topo -m 2>&1 | head -25

echo "===== 4. python / torch ====="
which python3; python3 -V
python3 - <<'PY' 2>&1 | head -40
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.hip:", torch.version.hip)
    print("torch.version.cuda:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name} vram={p.total_memory/2**30:.1f}GB cu={getattr(p,'multi_processor_count','?')} arch={getattr(p,'gcnArchName','?')}")
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    y = (x @ x).float().abs().mean().item()
    print("bf16 matmul OK, mean:", y)
    x = torch.randn(512, 512, device="cuda")
    print("fp32 matmul OK:", (x @ x).abs().mean().item())
except Exception as e:
    import traceback; traceback.print_exc()
PY

echo "===== 5. installed llm stacks ====="
for py in python3 /usr/bin/python3; do
  $py - <<'PY' 2>/dev/null
import importlib
for m in ["vllm","sglang","transformers","triton","vllm_plugin_das","sglang_das","flash_attn"]:
    try:
        mod=importlib.import_module(m); print(f"  {m}: {getattr(mod,'__version__','?')}")
    except Exception as e: print(f"  {m}: MISSING ({type(e).__name__})")
PY
done
echo "--- pip list head ---"
python3 -m pip list 2>/dev/null | grep -iE "^(torch|triton|vllm|sglang|transformers|flash|das)" | head -20

echo "===== 6. container runtime ====="
which apptainer singularity docker podman enroot 2>/dev/null
apptainer --version 2>/dev/null

echo "===== 7. storage ====="
df -h /public/home/luql /public/ai_data 2>/dev/null | tail -3
du -sh /public/home/luql 2>/dev/null
echo "--- ai_data models readable? ---"
ls /public/ai_data/models 2>&1 | head -3
echo "--- images ---"
ls -la /public/ai_data/images 2>/dev/null | grep -iE "sglang|vllm" | head -10

echo "===== 8. network (multi-node feasibility) ====="
ip -o link show 2>/dev/null | awk '{print $2, $17}' | head -10

echo "===== DONE ====="
