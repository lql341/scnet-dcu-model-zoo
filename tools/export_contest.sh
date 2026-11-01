#!/usr/bin/env bash
# 把 monorepo 里的一个条目导出成独立参赛仓库（结构完全符合赛题要求）
#
# 用法：
#   ./tools/export_contest.sh models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang
#   ./tools/export_contest.sh <条目目录> --remote git@github.com:lql341/xxx.git
#   ./tools/export_contest.sh <条目目录> --dry-run
set -euo pipefail
SRC="${1:?用法: $0 <条目目录> [--remote URL] [--dry-run]}"
shift || true
REMOTE=""; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --remote) REMOTE="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
done
[ -d "$SRC" ] || { echo "目录不存在: $SRC"; exit 1; }
NAME="$(basename "$SRC")"
OUT="/tmp/export-$(date +%s)-$NAME"

echo "[export] $NAME → $OUT"
mkdir -p "$OUT"
# 摊平到仓库根（赛题要求 README.md / env.sh / run.sh ... 在根目录）
cp -rf "$SRC"/. "$OUT"/
rm -rf "$OUT/.git" "$OUT/logs/.ipynb_checkpoints" 2>/dev/null || true

# 补 .gitignore（单件仓库不该忽略 logs/bench，它们是交付物）
cat > "$OUT/.gitignore" <<'GI'
*.tar
__pycache__/
*.pyc
.DS_Store
GI

# README 里若还残留占位 URL，提示
if grep -q '【URL 待填】' "$OUT/README.md" 2>/dev/null; then
  echo "[export] ! README 里仍有【URL 待填】，推送前请填好"
fi

echo "[export] 内容："
( cd "$OUT" && find . -type f ! -path './.git/*' | sort | sed 's/^/  /' )

if [ "$DRY" = "1" ]; then echo "[export] --dry-run，未创建 git 仓库"; exit 0; fi

( cd "$OUT"
  git init -q
  git add -A
  git -c user.email="$(git config --get user.email || echo dev@local)" \
      -c user.name="$(git config --get user.name || echo dev)" \
      commit -qm "feat: $NAME —— 单件快照（由 scnet-dcu-model-zoo 导出）"
  git branch -M main
  if [ -n "$REMOTE" ]; then
    git remote add origin "$REMOTE"
    git push -u origin main
    echo "[export] 已推送到 $REMOTE"
  else
    echo "[export] 仓库已就绪：$OUT"
    echo "[export] 推送可用：cd $OUT && git remote add origin <URL> && git push -u origin main"
  fi
)
