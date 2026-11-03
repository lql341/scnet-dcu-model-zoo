#!/bin/bash -l
# 镜像探针：Harbor 上现成的 BW 版 MiniMax-Music3 + SGLang 镜像里到底有什么
#   镜像 dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1（digest sha256:20f071aa…，压缩 12.27 GB）
# 目的：找出官方 `scripts/end_to_end/minimax_ttm_test.py` 依赖的那个 SGLang-Omni HTTP 服务的启动方式
# 用法：sbatch sweeps/music3-image-probe.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 00:40:00
#SBATCH -J music3-probe
#SBATCH -o /public/home/scnethr2601a10/work/music3-probe-%j.out
#SBATCH -e /public/home/scnethr2601a10/work/music3-probe-%j.err
set -uo pipefail
IMG=image.ac.com:5000/dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "[host] node=$(hostname) tmp_free=${FREE}G"
[ "$FREE" -lt 20 ] && { echo "[host] NOT_ENOUGH_SPACE exit 42"; exit 42; }
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/tmp/pull.log 2>&1 || { tail -3 /tmp/pull.log; exit 1; }
echo "[host] image ready $(date +%T)"

$PD run --rm --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri -v /opt/hyhal:/opt/hyhal:ro \
  "$IMG" -c '
echo "=== / ==="; ls /
echo "=== /opt ==="; ls /opt 2>/dev/null
echo "=== /workspace ==="; ls /workspace 2>/dev/null | head
echo "=== sglang 相关包 ==="; pip list 2>/dev/null | grep -iE "sglang|omni|music|minimax|diffusers|transformers|torch" | head -20
echo "=== python 版本 ==="; python3 -V
echo "=== sglang_omni 模块 ==="; python3 -c "import sglang_omni, os; print(os.path.dirname(sglang_omni.__file__))" 2>&1 | tail -2
echo "=== 可能的启动入口 ==="
for m in sglang_omni.cli sglang_omni.serve sglang_omni.server sglang_omni.launch; do
  python3 -c "import importlib,sys; importlib.import_module(\"$m\"); print(\"OK $m\")" 2>/dev/null
done
which sglang-omni sgl-omni sgo 2>/dev/null
echo "=== 镜像里的脚本/文档 ==="
find / -maxdepth 3 -name "*.md" -newer /etc/hostname 2>/dev/null | head -10
find / -maxdepth 3 \( -name "*.sh" -o -name "*.py" \) -path "*music*" 2>/dev/null | head -20
echo "=== pip show sglang-omni ==="; pip show sglang-omni 2>/dev/null | head -8
' 2>&1 | grep -viE "subuid|rootless|level=warning"
echo "[host] DONE $(date +%T)"
