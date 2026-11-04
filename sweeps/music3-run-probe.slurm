#!/bin/bash -l
# MiniMax-Music3 探针：用镜像里自带的 BW 适配工作区（/tmp/music3-hygon，我们已捞到 ~/work/music3-hygon）
# 起服务 → 跑官方条件前 2 条 → 检查是否真的出声、规格是否 32 kHz 立体声
#
# 背景与风险：这套 harness 是**镜像作者的个人工作区**，其最后一版日志里有 HW queue VMFault + core dump，
# 所以本轮只验「能不能出声」，不做任何结果声明。
# 提交：sbatch sweeps/music3-run-probe.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -J music3-run
#SBATCH -o /public/home/scnethr2601a10/work/music3-run-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/music3-run-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1
MODELHOST=/public/home/scnethr2601a10/models/MiniMax-Music3
HARNESS=/public/home/scnethr2601a10/work/music3-hygon
WORK=/public/home/scnethr2601a10/work/music3-run
PROBE=${PROBE_ONLY:-2}
mkdir -p "$WORK"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 60 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }

$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1
$PD image exists "$IMG" || { echo "[host] image unavailable"; tail -3 /tmp/pull.log; exit 1; }
echo "[host] image ready $(date +%T)"

# 官方条件清单：由 submit 侧准备好（~/work/official-conds/music3.jsonl ← 仓库 conditions/music3.jsonl）
COND_DIR=/public/home/scnethr2601a10/work/official-conds
[ -s "$COND_DIR/music3.jsonl" ] || { echo "[host] 缺 $COND_DIR/music3.jsonl"; exit 3; }
mkdir -p "$WORK/out"

echo "[host] run $(date +%T)"
$PD run --rm --ipc=host --privileged --security-opt seccomp=unconfined --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$MODELHOST":/models/Music3:ro \
  -v "$HARNESS":/work/harness -v "$WORK":/work/out -v "$COND_DIR":/work/conditions:ro \
  -e MODEL_PATH=/models/Music3 -e PORT=8000 -e PROBE_ONLY="$PROBE" \
  -e RUNTIME_PATH=/work/harness/runtime -e LOAD_TIMEOUT=1500 \
  "$IMG" -c '
set -uo pipefail
[ -f /opt/dtk/env.sh ] && source /opt/dtk/env.sh
export LD_LIBRARY_PATH=/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
cd /work/harness || exit 1
# 镜像里带出来的 runtime/ 里是作者环境里的**悬空符号链接**，直接复用会
# FileExistsError（runtime/backbone/generation_config.json 已存在）→ 先清空重建
rm -rf runtime logs .venv server.pid server.log __pycache__ music3_hygon/__pycache__ 2>/dev/null
mkdir -p runtime logs
source ./env.sh
export MODEL_PATH=/models/Music3 RUNTIME_PATH=/work/harness/runtime PORT=8000
echo "--- env_check ---"
bash env_check.sh 2>&1 | tail -12 || true
echo "--- 启动服务（run.sh 会打 patch、建 venv、轮询 /health）---"
bash run.sh 2>&1 | tail -8 || { echo "[inner] run.sh 失败"; tail -30 server.log 2>/dev/null; exit 1; }
echo "--- 条件数 ---"; wc -l /work/conditions/music3.jsonl 2>/dev/null || ls /work/conditions
echo "--- 评测前 ${PROBE_ONLY} 条 ---"
LIMIT_ARG=""
[ "${PROBE_ONLY}" != "0" ] && LIMIT_ARG="--limit ${PROBE_ONLY}"
python3 evaluate.py --conditions /work/conditions/music3.jsonl --output-root /work/out/music3 \
  --port 8000 ${LIMIT_ARG} 2>&1 | tail -20
echo "--- 产物 ---"
ls -la /work/out/music3/audio 2>/dev/null | head -5
python3 - <<PYEOF
import glob, wave, os
for f in sorted(glob.glob("/work/out/music3/audio/*.wav")):
    with wave.open(f) as w:
        print(os.path.basename(f), "sr=%d ch=%d frames=%d dur=%.2fs bytes=%d" % (
            w.getframerate(), w.getnchannels(), w.getnframes(),
            w.getnframes()/w.getframerate(), os.path.getsize(f)))
PYEOF
pkill -f serve.py 2>/dev/null
' 2>&1 | grep -viE "subuid|rootless|level=warning|^Copying|^Getting"
echo "[host] DONE $(date +%T)"
