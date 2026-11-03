#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:08:00
#SBATCH -J hrprobe6
#SBATCH -o /public/home/scnethr2601a10/probe/p6-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/p6-%j.err

module purge; module load compiler/gcc/9.3.0 compiler/dtk/26.04 2>/dev/null
echo "== node =="; hostname
echo "== local disk =="
df -h / /tmp /var/tmp /dev/shm 2>/dev/null | awk 'NR==1||/dev|tmp|shm|overlay|scnet/'
echo "== node local writable? =="
for d in /tmp /var/tmp /dev/shm; do touch $d/.w$$ 2>/dev/null && echo "$d OK" || echo "$d NO"; rm -f $d/.w$$ 2>/dev/null; done
echo "== home quota =="
df -h /public/home/scnethr2601a10 | tail -1
du -sh /public/home/scnethr2601a10 2>/dev/null
echo "== podman rootless =="
export XDG_RUNTIME_DIR=/public/home/scnethr2601a10/.run
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
podman info --format '{{.Store.GraphRoot}} | {{.Store.GraphDriverName}} | {{.Store.GraphRoot}}' 2>&1 | grep -v level= | tail -2
echo "== image tars readable =="
ls -la /public/ai_data/images/vllm018-ubuntu22.04-dtk26.04-gemma4-0413.tar 2>&1
echo "== DONE =="
