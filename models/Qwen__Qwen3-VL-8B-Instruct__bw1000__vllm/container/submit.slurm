#!/bin/bash -l
# BW1000 × vLLM × Qwen3-VL-8B-Instruct（T02 视觉多模态理解）
# 单镜像编排（2026-11-22 起）：本分区节点 /tmp 放不下两个镜像（gotcha #62）
#   容器 A  vLLM 镜像   起服务（--network=host），并在其中跑 env_check（需要 torch）
#   宿主机              跑 smoke.sh / verify.py（只需 curl + python3 标准库）
#   容器 B  vLLM 镜像   跑 bench_vllm.sh（`vllm bench serve`，不再是 SGLang 客户端）
#
# 用法（本机经 OpenAPI 提交，集群无 SSH/rsync 通道）：
#   bash container/submit.slurm                     # 官方 6 档基准
#   SKIP_BENCH=1 bash container/submit.slurm        # 只验到 smoke，跳过压测
#   BENCH_QUICK=1 bash container/submit.slurm       # 压测只跑 4K/并发1/1轮（自检用）
#
# 产物落在宿主机 $WORK/repo（env.log / server.log / smoke.log / bench/）
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 16
#SBATCH -t 03:00:00
#SBATCH -J qwenvl-dcu

set -uo pipefail

# sbatch 会把脚本拷到 spool 目录，$0 不可用；用 SLURM_SUBMIT_DIR 定位仓库根
REPO="${SLURM_SUBMIT_DIR:-$PWD}"
[ -f "$REPO/env.sh" ] || { echo "[host] 请在仓库根目录提交（找不到 $REPO/env.sh）"; exit 1; }
source "$REPO/env.sh"

WORK="${WORK:-/public/home/luql/work/qwenvl8b}"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" "$WORK"; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
echo "[host] node=$(hostname) tmp_free=${FREE}G  repo=$REPO  work=$WORK"
# 空间判据：vLLM 镜像 load 后约占 30 GB，留 40 GB 余量即可。
# （历史上曾同时加载 SGLang CLI 镜像跑压测，需 ≥145 GB，各节点 21–141 GB 波动等不到；
#   现改为同镜像内 `vllm bench serve`，只需一个镜像 —— gotcha #62。）
NEED=40
if [ "$FREE" -lt "$NEED" ]; then
  echo "[host] tmp_free=${FREE}G < need ${NEED}G，最多等 30 分钟等别的作业释放存储"
  for i in $(seq 1 60); do
    sleep 30
    FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
    [ "$FREE" -ge "$NEED" ] && { echo "[host] tmp_free=${FREE}G 达标（等了 $((i/2)) 分钟）"; break; }
    [ "$i" -eq 60 ] && { echo "[host] 等了 30 分钟仍不足（${FREE}G），exit 42 换节点"; exit 42; }
  done
fi
df -Pm /tmp | tail -1 | sed 's/^/[host] tmp before load: /'

# 镜像获取顺序：本地 tar.gz / tar → Harbor pull。
# 2026-11-22 实测 Harbor（image.ac.com:5000）在本集群**不可达**（connect timeout / no route to host），
# 且 vllm 0.21 仓库曾报 artifact not found。所以本地归档是主路径，pull 只作兜底。
# ⚠️ 旧版这里读的是 /tmp/load-$tag.log（缺 -$JID），失败时 tail 一个不存在的文件，
# 真错误被吞掉——本条目已修正为 $LOG。
# 只加载 vLLM 镜像：压测已改为同镜像内 `vllm bench serve`，不再需要 SGLang CLI 镜像（gotcha #62）。
SPECS="vLLM|$IMAGE|${HOST_IMAGE_TAR:-}"
while IFS='|' read -r tag img tar; do
  [ -z "${tag:-}" ] && continue
  LOG=/tmp/load-$tag-$JID.log
  echo "[host] load $tag image  (tar=${tar:-none})"
  if [ -n "$tar" ] && [ -s "$tar" ]; then
    case "$tar" in
      *.gz) gunzip -c "$tar" | $PD load 2>&1 | tail -3 ;;   # gunzip 流式，不占双份磁盘
      *)    $PD load -i "$tar" 2>&1 | tail -3 ;;
    esac
  fi
  if $PD image exists "$img" 2>/dev/null; then
    echo "[host] $tag image ready (from tar)"
  else
    echo "[host] $tag 不在本地，尝试 Harbor pull（本集群大概率不可达）"
    $PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$img" >"$LOG" 2>&1
    if $PD image exists "$img" 2>/dev/null; then
      echo "[host] $tag image ready (from Harbor)"
    else
      echo "[host] $tag IMAGE_UNAVAILABLE"
      tail -5 "$LOG" 2>/dev/null || echo "  (无 pull 日志)"
      echo "[host] 提示：Harbor 不可达或镜像已被删除。本地兜底归档在"
      echo "        /public/home/luql/work/imgstore/vllm021.tar（vLLM）"
      echo "        /public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz（CLI）"
      exit 1
    fi
  fi
done <<< "$SPECS"

# 仓库拷进 $WORK/repo（容器写产物后留在宿主机）
# OpenAPI 通道下本脚本就是从 $WORK/repo 启动的（REPO == WORK/repo），此时跳过拷贝：
# 把目录 rsync 到自己会重复嵌套（gotchas #33）。
mkdir -p "$WORK/repo"
if [ "$(readlink -f "$REPO")" != "$(readlink -f "$WORK/repo")" ]; then
  rsync -a --exclude bench --exclude official-eval "$REPO"/ "$WORK/repo"/ 2>/dev/null
else
  echo "[host] REPO 即 $WORK/repo，跳过自拷"
fi
chmod -R a+rwX "$WORK/repo" 2>/dev/null
# 收容器前先把服务端日志落盘，否则 500/崩溃的根因随容器消失（踩过：只剩 1 行 digest）
cleanup() { $PD logs vllm-srv > "$WORK/vllm-logs-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }
trap cleanup EXIT

# ---------- 容器 A：起服务（前台，detached） ----------
echo "[host] start vLLM server container $(date +%T)"
$PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
  --entrypoint=/bin/bash \
  --device /dev/kfd --device /dev/mkfd --device /dev/dri \
  -v /opt/hyhal:/opt/hyhal:ro -v "$HOST_MODEL_PATH":/models/M:ro \
  -v "$WORK":/work -v "$REPO":/repo:ro \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && bash env_check.sh > env.log 2>&1 && cp env.log /work/repo/env.log
source ./env.sh
exec python3 -m vllm.entrypoints.openai.api_server \
  --model \$MODEL_PATH --served-model-name \$SERVED_NAME \
  --host 127.0.0.1 --port \$PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len \$MAX_LEN --gpu-memory-utilization \$MEM_FRAC \
  --tensor-parallel-size \$TP --max-num-seqs \$MAX_NUM_SEQS
" > "$WORK/server-$JID.log" 2>&1

OK=0
for t in $(seq 1 240); do
  curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && { OK=1; echo "[host] server up after $((t*10))s $(date +%T)"; break; }
  $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || {
    echo "[host] container gone；完整日志 → $WORK/srvlog-$JID.txt"
    $PD logs vllm-srv > "$WORK/srvlog-$JID.txt" 2>&1
    grep -aE "Error|error|assert|Assertion|ValueError|TypeError|KeyError|No module|not supported|out of memory|No available|world_size|device" "$WORK/srvlog-$JID.txt" | grep -avE "core_client|contextlib|async_llm|otel" | head -20
    exit 1; }
  sleep 10
done
[ $OK -eq 1 ] || { echo "[host] SERVER_NOT_UP"; exit 1; }

# ---------- 宿主机：smoke + verify（无需 GPU/python 依赖） ----------
cd "$WORK/repo"
echo "[host] smoke $(date +%T)"
bash smoke.sh || echo "[host] smoke 有失败项，见 smoke.log"
echo "[host] verify selfcheck + verify"
python3 verify.py --selfcheck || true
python3 verify.py || echo "[host] verify.py 被官方基线文件阻塞，见 MISSING.md"

# ---------- 容器 B：压测（同镜像内 vllm bench serve） ----------
if [ "${SKIP_BENCH:-0}" = "1" ]; then
  echo "[host] SKIP_BENCH=1，跳过压测"
else
  echo "[host] bench（同镜像内 vllm bench serve，不再加载 CLI 镜像）$(date +%T)"
  # 为什么不用 CLI 镜像：本集群 Harbor 不可达，两个镜像共存需 ≥145 GB 节点空间，
  # 实测各节点在 21–141 GB 间波动，等不到（gotcha #62）。vLLM 镜像自带 `vllm bench serve`，
  # 在同一个容器里对 127.0.0.1 发压即可，省掉 20 GB。
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work -v "$REPO":/repo:ro \
    -v "$REPO/preflight_lib.py":/preflight_lib.py:ro \
    -v "$HOST_MODEL_PATH":/models/M:ro \
    -e BENCH_QUICK="${BENCH_QUICK:-0}" \
    "$IMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo && source ./env.sh && bash bench_vllm.sh
" 2>&1 | grep -viE "subuid|rootless single|Network file|level=warning"
fi

echo "[host] done $(date +%T)"
echo "[host] 产物目录：$WORK/repo"
ls -la "$WORK/repo"/env.log "$WORK/repo"/smoke.log "$WORK/repo"/bench/summary.md 2>/dev/null
