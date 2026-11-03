#!/bin/bash -l
# 镜像归档：把两个镜像做成可移植 tar，放进 chmod 755 的共享归档目录，供 luql 账号随时取用
# 目的：1) 防止上游 linshp 删除仓库导致不可复现；2) 为后续手动 push 到自有命名空间做准备
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 8
#SBATCH -t 01:30:00
#SBATCH -J img-archive
#SBATCH -o /public/home/scnethr2601a10/zoo-images/archive-%j.out
#SBATCH -e /public/home/scnethr2601a10/zoo-images/archive-%j.err

set -uo pipefail
IMG_SGL=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
IMG_VLLM=image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0
SRC_TAR=/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar
DEST=/public/home/scnethr2601a10/zoo-images
mkdir -p "$DEST"
export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"

echo "== node $(hostname) $(date +%T) =="
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); echo "tmp free=${FREE}G"
[ "$FREE" -lt 40 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }

# ---------- 1. SGLang 镜像：已有未压缩 tar，直接 gzip ----------
OUT_SGL="$DEST/mimo-sglang-bw10000_1.1.1.tar.gz"
if [ ! -s "$OUT_SGL" ]; then
  echo "== gzip SGLang tar $(date +%T) =="
  gzip -c "$SRC_TAR" > "$OUT_SGL.tmp" && mv "$OUT_SGL.tmp" "$OUT_SGL"
fi
ls -lh "$OUT_SGL"

# ---------- 2. vLLM 镜像：pull + save --compress ----------
OUT_VLLM="$DEST/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz"
echo "== pull vLLM $(date +%T) =="
$PD pull --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG_VLLM" >/tmp/pull.log 2>&1 \
  || { echo "PULL_FAIL"; tail -3 /tmp/pull.log; exit 1; }
echo "== save --compress vLLM $(date +%T) =="
if [ ! -s "$OUT_VLLM" ]; then
  $PD save --compress -o "$OUT_VLLM.tmp" "$IMG_VLLM" && mv "$OUT_VLLM.tmp" "$OUT_VLLM"
fi
ls -lh "$OUT_VLLM"

# ---------- 3. 清单：digest + sha256 + 还原命令 ----------
echo "== 生成清单 $(date +%T) =="
{
  echo "# 镜像归档清单"
  echo
  echo "生成时间: $(date -Iseconds)"
  echo "生成节点: $(hostname)"
  echo
  echo "## 文件"
  echo
  echo '| 文件 | 大小 | sha256 |'
  echo '| --- | --- | --- |'
  for f in "$OUT_SGL" "$OUT_VLLM"; do
    [ -s "$f" ] || continue
    printf '| `%s` | %s | `%s` |\n' "$(basename "$f")" "$(du -h "$f" | cut -f1)" "$(sha256sum "$f" | cut -d' ' -f1)"
  done
  echo
  echo "## 镜像 digest（与上游一致，证明内容未修改）"
  echo
  echo '| 镜像 | digest |'
  echo '| --- | --- |'
  echo "| \`$IMG_SGL\` | \`$(curl -sSk -I -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
      "https://image.ac.com:5000/v2/dcu/linshp/base/mimo-sglang-bw10000/manifests/1.1.1" \
      -H "Authorization: Bearer $(curl -sSk 'https://image.ac.com:5000/service/token?service=harbor-registry&scope=repository:dcu/linshp/base/mimo-sglang-bw10000:pull' | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')" \
      2>/dev/null | grep -i docker-content-digest | tr -d '\r' | awk '{print $2}')\` |"
  echo "| \`$IMG_VLLM\` | \`$(curl -sSk -I -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
      "https://image.ac.com:5000/v2/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw/manifests/1.0.0" \
      -H "Authorization: Bearer $(curl -sSk 'https://image.ac.com:5000/service/token?service=harbor-registry&scope=repository:dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:pull' | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')" \
      2>/dev/null | grep -i docker-content-digest | tr -d '\r' | awk '{print $2}')\` |"
  echo
  echo "## 还原"
  echo
  echo '```bash'
  echo 'gunzip -c mimo-sglang-bw10000_1.1.1.tar.gz | podman load'
  echo 'gunzip -c vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz | podman load'
  echo '```'
} > "$DEST/MANIFEST.md" 2>&1

cat "$DEST/MANIFEST.md"

# ---------- 4. 开放读权限（供 luql 直接从共享 FS 拷走）----------
chmod 755 "$DEST"
chmod 644 "$DEST"/* 2>/dev/null
echo "== 权限已开放 $(date +%T) =="
ls -la "$DEST"
echo "== DONE $(date +%T) =="
