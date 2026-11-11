#!/bin/bash -l
# 把 3 个镜像推到自有命名空间（Harbor: image.ac.com:5000）
#
# 前置：
#   1) 凭据文件存在：~/.harbor-pass（chmod 600），内容为 Harbor 密码
#      获取路径（按成功率）：
#        a. 个人中心 → 访问控制（拿 AK/SK 那页）→ 找「镜像仓库 / 容器镜像 / Harbor / 仓库密码」
#        b. scnet.cn/ui/aihub/image → 任一镜像详情 → 「使用此镜像」区块的 podman login 提示
#        c. 直接用平台账号密码试（Harbor 常接 LDAP/SSO）
#   2) 镜像已在本地（从归档 load，见 docs/manual-push.md 路径 B）
#
# 命名（实测同批账号的惯例）：
#   image.ac.com:5000/dcu/<账号>/base/<name>:<tag>     个人命名空间
#   image.ac.com:5000/aihub/dcu/<账号>/<name>:<tag>    平台镜像库可见（比赛用这个）
#
# 用法：
#   ACCOUNT=luql sbatch tools/push_harbor.sh
#   只推某一个：ONLY=m3 ACCOUNT=luql sbatch tools/push_harbor.sh
#   只看命令不执行：DRY_RUN=1 bash tools/push_harbor.sh
set -uo pipefail
ACCOUNT="${ACCOUNT:-luql}"
HARBOR="${HARBOR:-image.ac.com:5000}"
ARCHIVE="${ARCHIVE:-/public/home/luql/archive/zoo-dcu-2026-11/images}"
ONLY="${ONLY:-}"
DRY_RUN="${DRY_RUN:-0}"

export XDG_RUNTIME_DIR=/tmp/xdg-$$
mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"
PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
TLS=(--tls-verify=false)

# 镜像清单：归档文件 | 原镜像名 | 目标短名
IMAGES=(
  "mimo-sglang-bw10000_1.1.1.tar.gz|dcu/linshp/base/mimo-sglang-bw10000:1.1.1|mimo-sglang-bw10000:1.1.1"
  "vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz|dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0|vllm0.20-qwen_asr1.7-bw:1.0.0"
  "minimax-music3-bw-sglang_20261031-rc1.tar.gz|dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1|minimax-music3-bw-sglang:20261031-rc1"
)

log() { echo "[$(date +%T)] $*"; }

if [ "$DRY_RUN" != "1" ]; then
  [ -s "$HOME/.harbor-pass" ] || { echo "[预检] 缺 ~/.harbor-pass，见本脚本顶部说明"; exit 2; }
  log "登录 $HARBOR（账号 $ACCOUNT）"
  $PD login "$HARBOR" -u "$ACCOUNT" --password-stdin "${TLS[@]}" < "$HOME/.harbor-pass" \
    || { echo "[预检] 登录失败：账号或密码不对"; exit 2; }
fi

FAIL=0
for spec in "${IMAGES[@]}"; do
  IFS='|' read -r FILE SRC SHORT <<<"$spec"
  case "$ONLY" in
    "") ;;
    m3|music3) [ "${FILE#minimax}" = "$FILE" ] && continue ;;
    sgl|sglang) [ "${FILE#mimo}" = "$FILE" ] && continue ;;
    vllm) [ "${FILE#vllm}" = "$FILE" ] && continue ;;
    *) [ "${FILE#$ONLY}" = "$FILE" ] && continue ;;
  esac
  NAME="${SHORT%%:*}"; TAG="${SHORT##*:}"
  D1="$HARBOR/dcu/$ACCOUNT/base/$NAME:$TAG"
  D2="$HARBOR/aihub/dcu/$ACCOUNT/$NAME:$TAG"

  echo
  log "== $NAME:$TAG =="
  if [ "$DRY_RUN" = "1" ]; then
    echo "  gunzip -c $ARCHIVE/$FILE | \$PD load"
    echo "  \$PD tag $SRC $D1 && \$PD push $D1"
    echo "  \$PD tag $SRC $D2 && \$PD push $D2"
    continue
  fi

  # 1) 本地有就跳过 load
  if ! $PD image exists "$SRC" 2>/dev/null; then
    [ -s "$ARCHIVE/$FILE" ] || { log "  缺归档 $ARCHIVE/$FILE"; FAIL=$((FAIL+1)); continue; }
    log "  load 归档（$(du -h "$ARCHIVE/$FILE" | cut -f1)）"
    gunzip -c "$ARCHIVE/$FILE" | $PD load >/tmp/load-$JID-$NAME.log 2>&1 \
      || { log "  load 失败"; tail -3 /tmp/load-$JID-$NAME.log; FAIL=$((FAIL+1)); continue; }
  else
    log "  本地已有镜像"
  fi
  SRC_DIGEST=$($PD image inspect "$SRC" --format '{{.Digest}}' 2>/dev/null)

  # 2) tag + push 两个目标
  for DST in "$D1" "$D2"; do
    $PD tag "$SRC" "$DST" || { log "  tag 失败 $DST"; FAIL=$((FAIL+1)); continue; }
    log "  push → $DST"
    if $PD push "${TLS[@]}" "$DST" >/tmp/push-$JID-$NAME.log 2>&1; then
      NEW_DIGEST=$($PD image inspect "$DST" --format '{{.Digest}}' 2>/dev/null)
      log "   ✅ push 成功  digest=$NEW_DIGEST"
      [ -n "$SRC_DIGEST" ] && [ "$NEW_DIGEST" != "$SRC_DIGEST" ] && \
        log "   ⚠ 本地 digest 与原镜像不同（$SRC_DIGEST）—— 检查是否被重建"
    else
      log "   ✗ push 失败"; tail -5 /tmp/push-$JID-$NAME.log | sed 's/^/     /'
      FAIL=$((FAIL+1))
    fi
  done
  # 3) 释放本地存储，避免 /tmp 撑爆
  $PD rmi -f "$D1" "$D2" >/dev/null 2>&1
done

echo
log "完成，失败 $FAIL 项"
log "验证（匿名即可）："
log "  curl -sk '$HARBOR/api/v2.0/projects/aihub/repositories?page_size=100&q=name%3D~$ACCOUNT' | jq -r '.[].name'"
