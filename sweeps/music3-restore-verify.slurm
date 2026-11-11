#!/bin/bash -l
# P1-10：Music3 镜像归档的**还原验证**（不是只算 sha256）：从 tar.gz load → 起 harness 服务 → 跑 1 条官方条件
#   sbatch sweeps/music3-restore-verify.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J m3-restore
#SBATCH -o /public/home/scnethr2601a10/work/m3-restore-%j.out
set -uo pipefail
TARGZ=/public/home/scnethr2601a10/zoo-images/minimax-music3-bw-sglang_20261031-rc1.tar.gz
IMG=image.ac.com:5000/dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1
MODELHOST=/public/home/scnethr2601a10/models/MiniMax-Music3
HARNESS=/public/home/scnethr2601a10/work/music3-hygon
WORK=/public/home/scnethr2601a10/work/m3-restore
COND=/public/home/scnethr2601a10/work/official-conds/music3.jsonl
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
# 13.7 GB 归档（内含 14 GB core dump）解包 + podman 存储要双份空间，实测 83 G 仍失败 → 110 G
[ "$FREE" -lt 110 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
echo "[host] 从归档还原（gunzip | podman load）$(date +%T)"
gunzip -c "$TARGZ" | $PD load >/tmp/load-$JID.log 2>&1 || { echo "[host] load 失败"; tail -5 /tmp/load-$JID.log; exit 1; }
$PD images --format '{{.Repository}}:{{.Tag}}' | grep -i music3 || { echo "[host] 镜像名不符（可能被重建）"; exit 2; }
echo "[host] load OK $(date +%T)"
echo "[host] 跑 1 条官方条件验证功能"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  -v "$MODELHOST":/models/Music3:ro -v "$HARNESS":/work/harness -v "$WORK":/work/out \
  -v "$COND":/work/conditions/music3.jsonl:ro \
  -e MODEL_PATH=/models/Music3 -e PORT=8000 -e PROBE_ONLY=1 \
  "$IMG" -c '
set -uo pipefail
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
cd /work/harness || exit 1
rm -rf runtime logs .venv server.pid server.log __pycache__ music3_hygon/__pycache__ 2>/dev/null
mkdir -p runtime logs; source ./env.sh
export MODEL_PATH=/models/Music3 RUNTIME_PATH=/work/harness/runtime PORT=8000
bash run.sh 2>&1 | tail -3 || { echo "[inner] 服务启动失败"; tail -20 server.log; exit 3; }
LIMIT="--limit 1"
python3 evaluate.py --conditions /work/conditions/music3.jsonl --output-root /work/out/m3 \
  --port 8000 $LIMIT 2>&1 | tail -3
python3 - <<PYEOF
import glob, wave, os
fs = glob.glob("/work/out/m3/audio/*.wav")
print("[verify] 产物数:", len(fs))
for f in fs:
    with wave.open(f) as w:
        print("  %s sr=%d ch=%d dur=%.2fs bytes=%d" % (os.path.basename(f), w.getframerate(),
              w.getnchannels(), w.getnframes()/w.getframerate(), os.path.getsize(f)))
PYEOF
pkill -f serve.py 2>/dev/null
' 2>&1 | grep -viE "subuid|rootless|level=warning|^Copying|^Getting"
echo "[host] DONE $(date +%T)"
