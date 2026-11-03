#!/usr/bin/env bash
# 一次收尾：校验 → 重生成索引 → 提交 → 同步两个远端（GitHub 权威 + GitLab 镜像）
#
# 为什么必须两个远端都推：本项目第一原则是「镜像与经验 > 比赛提交」。
# GitHub 是权威积累库，GitLab（sourcefind）是比赛平台侧可读的镜像；
# 两处任何一处缺失，这次会话踩的坑就等于白踩。
#
# 用法：
#   tools/publish.sh "commit message"          # 校验 + 提交 + 推双远端
#   tools/publish.sh "msg" --no-push           # 只校验 + 提交
#   tools/publish.sh "" --check                # 只跑三件套校验，不提交
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
MSG="${1:-}"
MODE="${2:-}"

fail() { echo "[publish] ✗ $*" >&2; exit 1; }

echo "== 1/4 规范校验 =="
./tools/lint_entry.sh >/tmp/lint.out 2>&1 || { tail -20 /tmp/lint.out; fail "lint 未通过"; }
tail -1 /tmp/lint.out

echo "== 2/4 重生成索引 =="
python3 tools/gen_index.py || fail "gen_index 失败"

echo "== 3/4 体积守门 =="
./tools/check_quota.sh >/tmp/quota.out 2>&1 || { tail -20 /tmp/quota.out; fail "体积超限"; }
tail -1 /tmp/quota.out

[ "$MODE" = "--check" ] && { echo "[publish] 仅校验完成（工作区改动未提交）"; git status --short; exit 0; }

echo "== 4/4 提交并推双远端 =="
if [ -n "$(git status --porcelain)" ]; then
  [ -n "$MSG" ] || fail "工作区有改动但没给提交信息"
  git add -A && git commit -q -m "$MSG" || fail "commit 失败"
  echo "已提交：$(git log --oneline -1)"
else
  echo "工作区干净，跳过提交"
fi

[ "$MODE" = "--no-push" ] && { echo "[publish] 已跳过推送"; exit 0; }

git push -q origin HEAD || echo "[publish] ⚠ GitHub 推送失败（检查网络 / gh auth）"

# GitLab 远端：优先用 https + token，避免把 token 写进 .git/config
TOKEN_FILE="$HOME/.sourcefind-token"
GL="https://developer.sourcefind.cn/codes/luql/scnet-dcu-model-zoo.git"
if [ ! -f "$TOKEN_FILE" ]; then
  echo "[publish] ⚠ 缺 $TOKEN_FILE，跳过 GitLab"
else
  T=$(cat "$TOKEN_FILE")
  git push -q "https://oauth2:$T@developer.sourcefind.cn/codes/luql/scnet-dcu-model-zoo.git" HEAD:main \
    && echo "[publish] ✓ GitLab 已同步" \
    || echo "[publish] ⚠ GitLab 推送失败（main 是保护分支时不能 force push）"
fi
echo "[publish] 完成"
