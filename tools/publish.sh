#!/usr/bin/env bash
# 一次收尾：校验 → 重生成索引 → 提交 → 推 GitHub
#
# 本项目第一原则是「镜像与经验 > 比赛提交」，GitHub 是权威积累库。
# GitLab/sourcefind 远端已于 2026-11-21 按用户决定移除，不再推送。
#
# 用法：
#   tools/publish.sh "commit message"          # 校验 + 提交 + 推 GitHub
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

echo "== 4/4 提交并推 GitHub =="
if [ -n "$(git status --porcelain)" ]; then
  [ -n "$MSG" ] || fail "工作区有改动但没给提交信息"
  git add -A && git commit -q -m "$MSG" || fail "commit 失败"
  echo "已提交：$(git log --oneline -1)"
else
  echo "工作区干净，跳过提交"
fi

[ "$MODE" = "--no-push" ] && { echo "[publish] 已跳过推送"; exit 0; }

# GitHub：优先 SSH，失败则退回 gh 的 HTTPS 凭据（实测 SSH key 有时不可用：
#   git@github.com: Permission denied (publickey)，而 `gh auth status` 仍正常）
if ! git push -q origin HEAD 2>/dev/null; then
  echo "[publish] SSH 推送失败，改用 gh 凭据走 HTTPS"
  git -c credential.helper='!gh auth git-credential' push -q \
    https://github.com/lql341/scnet-dcu-model-zoo.git HEAD:main \
    || echo "[publish] ⚠ GitHub 推送失败（检查网络 / gh auth）"
fi
echo "[publish] 完成"
