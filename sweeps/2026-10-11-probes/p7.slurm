#!/bin/bash -l
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:08:00
#SBATCH -J hrprobe7
#SBATCH -o /public/home/scnethr2601a10/probe/p7-%j.out
#SBATCH -e /public/home/scnethr2601a10/probe/p7-%j.err

echo "== from COMPUTE node: reach registry? =="
timeout 8 curl -sSk -o /dev/null -w "image.ac.com:5000/v2/ => %{http_code}\n" https://image.ac.com:5000/v2/ 2>&1
timeout 8 bash -c 'cat < /dev/tcp/10.13.17.207/5000' >/dev/null 2>&1 && echo "tcp 5000 OPEN" || echo "tcp 5000 CLOSED"
timeout 8 curl -sS -o /dev/null -w "scnet.cn => %{http_code}\n" https://www.scnet.cn 2>&1

echo "== podman rootless =="
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p $XDG_RUNTIME_DIR && chmod 700 $XDG_RUNTIME_DIR
podman info 2>&1 | grep -viE "level=|subuid|rootless single|Network file" | grep -iE "graphRoot|graphDriverName|graphDriver|rootless" | head -6
echo "--- version ---"; podman --version

echo "== NFS read speed (1GB) =="
time dd if=/public/ai_data/images/vllm018-ubuntu22.04-dtk26.04-gemma4-0413.tar of=/dev/null bs=1M count=1024 2>&1 | tail -2

echo "== local nvme write speed (2GB) =="
time dd if=/dev/zero of=/tmp/speedtest bs=1M count=2048 oflag=direct 2>&1 | tail -2
rm -f /tmp/speedtest

echo "== DONE =="
