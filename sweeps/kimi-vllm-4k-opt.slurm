#!/bin/bash -l
# Kimi × vLLM：验证 4K/c1「每步 24 ms 固定气泡」是否可通过编译/调度参数改善
#
# 背景（2026-11-15 分析，见 docs/kimi-vllm-4k-c1-analysis.md）：
#   vLLM TPOT 24.18 ms 对 batch 与 context 都近乎不敏感（4K/c1 24.18 → 4K/c8 24.96 → 16K/c1 25.23），
#   拟合 = 24.07 ms + 0.111 ms/req；SGLang 为 7.64 + 1.67 ms/req。
#   → 每步先付 ~24 ms 固定开销 ≈ 27 层 × 0.89 ms/层，指向逐层 eager kernel launch + TP all-reduce。
#   机理线索：TritonMLABackend 的 AttentionCGSupport.NEVER 迫使 cudagraph_mode=PIECEWISE，
#   7 层 MLA 的 decode 注意力永远进不了整图，每步都要 eager 发射。
#
# 本作业先回答一个零成本问题：**该条目当前 compilation_config.mode 到底是什么**
#   - 若是 NONE  → 开 VLLM_COMPILE 有明确收益空间（E1 值得跑）
#   - 若是 VLLM_COMPILE → E1 作废，只跑 E2/E3
# 然后按结果依次跑变体。
#
# 成本：冷启动 640–700 s（实测），每个变体 4K/c1 + 4K/c8 约 8 min，合计约 20 min/变体。
#
# 用法：sbatch sweeps/kimi-vllm-4k-opt.slurm
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:4
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -J kimi-v4k
#SBATCH -o /public/home/luql/work/kimi-vllm-4k/out_%j.log

set -uo pipefail

# 注意：WORK 与 PSTORE 都在【家目录】（NFS，432G），不放节点本地 /tmp。
# 只有 podman 的镜像存储必须放节点本地——NFS 不支持 overlay（lsetxattr）也不支持 pivot_root（vfs）。
REPO=/public/home/luql/work/zoo-sync
WORK=/public/home/luql/work/kimi-vllm-4k
JID="${SLURM_JOB_ID:-$$}"
export XDG_RUNTIME_DIR=/tmp/xdg-$JID
mkdir -p "$XDG_RUNTIME_DIR" "$WORK/cache" "$WORK/repo" /tmp/emptycerts
chmod 700 "$XDG_RUNTIME_DIR"

# 镜像存储留在节点本地，但每次作业独立 root，并在退出时正规清理（避免 uid 映射残留删不掉）
PSTORE=/tmp/pr-$JID
PD="podman --root $PSTORE --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

IMGDIR=/public/home/luql/archive/zoo-dcu-2026-11/images
LIBPATH='/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/.hyhal/hsa/lib:/opt/dtk/.hyhal/hydm/lib:/opt/dtk/.hyhal/rocm_smi/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/usr/local/lib:/usr/local/lib/python3.10/dist-packages/lightop/lib'

MODEL_HOST=/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct
VIMAGE="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
VTAR="$IMGDIR/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw_1.0.0.tar.gz"
CIMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
CTAR="$IMGDIR/mimo-sglang-bw10000_1.1.1.tar.gz"
PORT=30000
MLEN=20480

log() { echo "[kv $(date +%H:%M:%S)] $*"; }
echo "[kv] node=$(hostname)"
echo "[kv] AllocTRES=$(sacct -j $JID -n -P -o AllocTRES 2>/dev/null | head -1)"
echo "[kv] WORK=$WORK (家目录) PSTORE=$PSTORE (节点本地)"

# ---- 磁盘自检：节点根盘水位极不稳（实测 0~278 GB），不足 45 GB 主动退出重投 ----
FREE_G=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}')
log "节点 /tmp 可用 ${FREE_G}G"
[ "$FREE_G" -ge 45 ] || { log "NOT_ENOUGH_SPACE: 需要 ≥45G，实际 ${FREE_G}G。exit 42 让调度器换节点"; exit 42; }

[ -d "$MODEL_HOST" ] || { log "模型目录不存在"; exit 1; }

load_image() {
  local img="$1" tar="$2" name="$3"
  $PD image exists "$img" 2>/dev/null && return 0
  log "load $name"
  gunzip -c "$tar" | $PD load >"$WORK/load-$name-$JID.log" 2>&1 \
    || { log "FAIL load $name"; tail -5 "$WORK/load-$name-$JID.log"; return 1; }
}

cleanup_all() {
  $PD logs vllm-srv >> "$WORK/allsrv-$JID.txt" 2>&1
  $PD rm -f vllm-srv >/dev/null 2>&1
  $PD rmi -af >/dev/null 2>&1
  $PD system prune -af >/dev/null 2>&1
  rm -rf "$PSTORE" "/tmp/prr-$JID" "$XDG_RUNTIME_DIR" 2>/dev/null
}
trap cleanup_all EXIT

kill_srv() { $PD logs vllm-srv >> "$WORK/allsrv-$JID.txt" 2>&1; $PD rm -f vllm-srv >/dev/null 2>&1; }

# $1=tag $2=额外参数（可含空格）
start_srv() {
  local tag="$1" extra="$2"
  log "启动 $tag : $extra"
  $PD run -d --name vllm-srv --network=host --privileged --security-opt seccomp=unconfined \
    --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$MODEL_HOST":/models/Kimi:ro \
    -v "$WORK":/work -v "$WORK/cache":/root/.cache/vllm \
    -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3 \
    "$VIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
export ROCM_PATH=/opt/dtk HIP_PATH=/opt/dtk/hip HYHAL_PATH=/opt/hyhal
cd /work/repo
exec python3 -m vllm.entrypoints.openai.api_server \
  --model /models/Kimi --served-model-name kimi-linear-48b \
  --host 127.0.0.1 --port $PORT --trust-remote-code --dtype bfloat16 \
  --max-model-len $MLEN --gpu-memory-utilization 0.92 \
  --tensor-parallel-size 4 --max-num-seqs 128 $extra
" > "$WORK/server-$tag-$JID.log" 2>&1
}

wait_srv() {   # $1=tag，就绪后把容器日志落到 $WORK/srv-<tag>-$JID.txt
  local tag="$1" t
  for t in $(seq 1 300); do
    if curl -sf --noproxy '*' -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
      $PD logs vllm-srv > "$WORK/srv-$tag-$JID.txt" 2>&1
      return 0
    fi
    $PD ps --format '{{.Names}}' 2>/dev/null | grep -q vllm-srv || return 1
    sleep 10
  done
  $PD logs vllm-srv > "$WORK/srv-$tag-$JID.txt" 2>&1
  return 1
}

# $1=ilen $2=conc $3=tag
run_bench() {
  local ilen="$1" conc="$2" tag="$3"
  rm -f "$WORK/repo/bench/"*.jsonl
  $PD run --rm --network=host --entrypoint=/bin/bash \
    --device /dev/kfd --device /dev/mkfd --device /dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro -v "$WORK":/work \
    -v "$REPO"/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm/preflight_lib.py:/preflight_lib.py:ro \
    -v "$MODEL_HOST":/models/Kimi:ro \
    -e BENCH_ILENS="$ilen" -e BENCH_CONCS="$conc" -e BENCH_ROUNDS="1" -e READY_TIMEOUT=900 \
    -e PORT=$PORT -e SERVED_NAME=kimi-linear-48b \
    -e LD_LIBRARY_PATH="$LIBPATH" -e ROCM_PATH=/opt/dtk -e HIP_PATH=/opt/dtk/hip -e HYHAL_PATH=/opt/hyhal \
    "$CIMAGE" -c "
export LD_LIBRARY_PATH=$LIBPATH:\$LD_LIBRARY_PATH
cd /work/repo && bash bench.sh
" 2>&1 | grep -viE "subuid|rootless single|level=warning" | tail -10
  [ -s "$WORK/repo/bench/round1.jsonl" ] && cp -f "$WORK/repo/bench/round1.jsonl" "$WORK/$tag-$JID.jsonl"
}

# 从服务端日志抓编译模式 —— 这是决定 E1 是否值得跑的关键信息
grab_mode() {   # $1=tag，从 podman logs 落盘的文件里抓 compilation mode
  local tag="$1" mode
  mode=$(grep -aoE "'mode': <CompilationMode\.[A-Z_]+" "$WORK/srv-$tag-$JID.txt" 2>/dev/null | head -1 | sed 's/.*CompilationMode\.//')
  if [ -z "$mode" ]; then
    mode=$(grep -aoE "CompilationMode\.[A-Z_]+" "$WORK/srv-$tag-$JID.txt" 2>/dev/null | head -1 | sed 's/.*CompilationMode\.//')
  fi
  echo "$mode"
}

: > "$WORK/matrix-$JID.txt"

load_image "$VIMAGE" "$VTAR" vllm020 || exit 1
load_image "$CIMAGE" "$CTAR" cli || { log "FAIL 客户端镜像，压测无法进行"; exit 1; }
rsync -a --exclude bench --exclude logs --exclude official-eval "$REPO"/models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm/ "$WORK/repo"/
chmod -R a+rwX "$WORK/repo"

# ============ 变体 E0：基线（同时抓 compilation mode）============
kill_srv
start_srv "E0_base" ""
if wait_srv "E0_base"; then
  MODE=$(grab_mode "E0_base")
  log "compilation_config.mode = ${MODE:-未抓到}"
  echo "MODE=${MODE:-unknown}" >> "$WORK/matrix-$JID.txt"
  for pair in "4096 1" "4096 8"; do
    set -- $pair
    log "  E0 跑 ${1}/c${2}"
    run_bench "$1" "$2" "E0_${1}_c${2}"
  done
  python3 - "$WORK" "E0_base" "$JID" >> "$WORK/matrix-$JID.txt" <<'PY'
import json,sys,glob
w,name,jid=sys.argv[1],sys.argv[2],sys.argv[3]
for p in sorted(glob.glob(f"{w}/{name}_*_c*_{jid}.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        d=json.loads(l)
        print("RESULT %s in=%d c%d ok=%d out=%.2f ttft=%.0f tpot=%.2f"
              %(name,d["random_input_len"],d["max_concurrency"],d["completed"],
                d["output_throughput"],d["mean_ttft_ms"],d["mean_tpot_ms"]))
PY
else
  log "E0 服务未起来"
  $PD logs vllm-srv > "$WORK/srv-E0-$JID.txt" 2>&1
  grep -aE "Error|assert|ValueError|NotImplemented|VMFault" "$WORK/srv-E0-$JID.txt" | head -8
  echo "RESULT E0_base SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
fi

# ============ 变体 E1：VLLM_COMPILE（仅当基线 mode 不是它时才跑）============
MODE=$(grep -oE "^MODE=.*" "$WORK/matrix-$JID.txt" | head -1 | cut -d= -f2)
if [ "${MODE:-unknown}" != "VLLM_COMPILE" ] && [ "${MODE:-unknown}" != "unknown" ]; then
  kill_srv
  start_srv "E1_compile" "--compilation-config '{\"mode\":\"VLLM_COMPILE\"}'"
  if wait_srv "E1_compile"; then
    log "E1 就绪"
    for pair in "4096 1" "4096 8"; do
      set -- $pair
      log "  E1 跑 ${1}/c${2}"
      run_bench "$1" "$2" "E1_${1}_c${2}"
    done
    python3 - "$WORK" "E1_compile" "$JID" >> "$WORK/matrix-$JID.txt" <<'PY'
import json,sys,glob
w,name,jid=sys.argv[1],sys.argv[2],sys.argv[3]
for p in sorted(glob.glob(f"{w}/{name}_*_c*_{jid}.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        d=json.loads(l)
        print("RESULT %s in=%d c%d ok=%d out=%.2f ttft=%.0f tpot=%.2f"
              %(name,d["random_input_len"],d["max_concurrency"],d["completed"],
                d["output_throughput"],d["mean_ttft_ms"],d["mean_tpot_ms"]))
PY
  else
    log "E1 服务未起来（该配置可能不被支持，属预期内）"
    $PD logs vllm-srv > "$WORK/srv-E1-$JID.txt" 2>&1
    echo "RESULT E1_compile SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
  fi
else
  log "基线 mode=${MODE:-unknown}，E1 跳过（已是 VLLM_COMPILE 或无信息）"
  echo "RESULT E1 SKIPPED mode=$MODE" >> "$WORK/matrix-$JID.txt"
fi

# ============ 变体 E2：async scheduling ============
kill_srv
start_srv "E2_async" "--async-scheduling"
if wait_srv "E2_async"; then
  log "E2 就绪"
  for pair in "4096 1" "4096 8"; do
    set -- $pair
    log "  E2 跑 ${1}/c${2}"
    run_bench "$1" "$2" "E2_${1}_c${2}"
  done
  python3 - "$WORK" "E2_async" "$JID" >> "$WORK/matrix-$JID.txt" <<'PY'
import json,sys,glob
w,name,jid=sys.argv[1],sys.argv[2],sys.argv[3]
for p in sorted(glob.glob(f"{w}/{name}_*_c*_{jid}.jsonl")):
    for l in open(p):
        if not l.strip(): continue
        d=json.loads(l)
        print("RESULT %s in=%d c%d ok=%d out=%.2f ttft=%.0f tpot=%.2f"
              %(name,d["random_input_len"],d["max_concurrency"],d["completed"],
                d["output_throughput"],d["mean_ttft_ms"],d["mean_tpot_ms"]))
PY
else
  log "E2 服务未起来"
  $PD logs vllm-srv > "$WORK/srv-E2-$JID.txt" 2>&1
  grep -aE "Error|unrecognized|assert|ValueError" "$WORK/srv-E2-$JID.txt" | head -6
  echo "RESULT E2_async SERVER_NOT_UP" >> "$WORK/matrix-$JID.txt"
fi

echo "==================== 汇总 ===================="
cat "$WORK/matrix-$JID.txt"

python3 - "$WORK" "$JID" <<'PY'
import re,sys
w,jid=sys.argv[1],sys.argv[2]
v={}
for line in open(f"{w}/matrix-{jid}.txt"):
    m=re.match(r"RESULT (\S+) in=(\d+) c(\d+) .*out=([\d.]+).*tpot=([\d.]+)",line)
    if m: v[(m.group(1),int(m.group(3)))]=(float(m.group(4)),float(m.group(5)))
base=v.get(("E0_base",1))
print()
for tag in ("E0_base","E1_compile","E2_async"):
    for c in (1,8):
        if (tag,c) in v:
            out,tpot=v[(tag,c)]
            delta=""
            if base and tag!="E0_base" and (tag,c) in v:
                delta=" (%+.1f%% vs E0)"%((out/base[0]-1)*100) if c==1 else ""
            print(f"{tag:<16} 4K/c{c}: out={out:7.2f}  tpot={tpot:6.2f} ms{delta}")
print()
if base:
    print(f"基线 TPOT(c1) = {base[1]:.2f} ms —— 若各变体 TPOT(c1) 仍 ≈ 24 ms，")
    print("  说明固定气泡来自 TritonMLA 无法整图捕获 + TP=4 逐层集合通信，参数改不动。")
PY
log "产物在 $WORK（家目录）"
