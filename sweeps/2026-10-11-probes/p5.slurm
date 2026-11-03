#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:8
#SBATCH -c 32
#SBATCH -t 00:10:00
#SBATCH -J hrprobe5
#SBATCH -o /public/home/scnethr2601a10/probe/p5-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/p5-%j.err

module purge
module load compiler/gcc/9.3.0 compiler/dtk/26.04 2>/dev/null

echo "===== node ====="; hostname; nproc
echo "===== hy-smi (8 cards) ====="
hy-smi 2>&1 | head -20
echo "===== vram per card ====="
hy-smi --showmeminfo vram 2>&1 | head -30
echo "===== hipInfo ====="
command -v hipInfo && hipInfo 2>&1 | grep -iE "Name|gcnArch|Total Global|Marketing|Device|SDMA|Managed" | head -20
echo "===== hipcc ====="
command -v hipcc && hipcc --version 2>&1 | head -3
echo "===== podman storage ====="
podman info --format '{{.Store.GraphRoot}} {{.Store.GraphDriverName}}' 2>&1 | head -2
df -h /public/home/scnethr2601a10 | tail -1
echo "===== ai_data images ====="
ls /public/ai_data/images/*.tar 2>/dev/null | wc -l
echo "===== DONE ====="
