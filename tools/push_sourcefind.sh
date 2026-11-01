#!/usr/bin/env bash
# 把两个 MiMo 仓库推到光合代码托管（GitLab CE，挂在 /codes 前缀下）
#
# 前置（只需做一次）：
#   1) 浏览器登录 https://developer.sourcefind.cn/codes
#   2) 右上头像 → Preferences → Access Tokens → 新建，勾选 scope: api + write_repository
#   3) 把 token 写进本机文件（不要贴到对话里）：
#        printf '%s' 'PASTE_TOKEN_HERE' > ~/.sourcefind-token && chmod 600 ~/.sourcefind-token
#
# 用法：
#   ./push_sourcefind.sh --check     # 只校验 token / 打印身份
#   ./push_sourcefind.sh --create    # 校验 token + 自动创建两个空项目
#   ./push_sourcefind.sh --push      # 校验 + 创建 + 推送
#
# token 不会写进 .git/config，也不会进 shell history（由 credential helper 从文件读取）。

set -euo pipefail

HOST=https://developer.sourcefind.cn
BASE="$HOST/codes"
API="$BASE/api/v4"
TOKEN_FILE="$HOME/.sourcefind-token"
ROOT="$(cd "$(dirname "$0")" && pwd)"

REPOS=(
  "kimi-repo:kimi-linear-48b-bw1000:BW1000 × SGLang 0.5.10 × Kimi-Linear-48B-A3B-Instruct (TP=4)"
  "bw-vllm-repo:mimo-vllm-bw1000:BW1000 × vLLM 0.20.0 × MiMo-V2.6-Distill-Qwen-9B"
  "mimo-repo:mimo-sglang-bw1000:BW1000 × SGLang 0.5.10 × MiMo-V2.6-Distill-Qwen-9B"
)

need_token() {
  cat <<'EOF'
[!] 找不到 ~/.sourcefind-token

请在你自己终端执行（把 PASTE_TOKEN_HERE 换成真实 token）：

  printf '%s' 'PASTE_TOKEN_HERE' > ~/.sourcefind-token && chmod 600 ~/.sourcefind-token

获取方式：浏览器登录 https://developer.sourcefind.cn/codes
  → 头像 → Preferences → Access Tokens → 新建
  → Scopes 勾选：api、write_repository
EOF
  exit 1
}

[ -s "$TOKEN_FILE" ] || need_token
TOKEN="$(tr -d '\r\n' < "$TOKEN_FILE")"

api() {  # api METHOD PATH [JSON]
  local m="$1" p="$2" d="${3:-}"
  if [ -n "$d" ]; then
    curl -sS -m 30 -X "$m" -H "PRIVATE-TOKEN: $TOKEN" -H "Content-Type: application/json" \
         -d "$d" "$API$p"
  else
    curl -sS -m 30 -X "$m" -H "PRIVATE-TOKEN: $TOKEN" "$API$p"
  fi
}

echo "== 1. 校验 token =="
USER_JSON="$(api GET /user || true)"
USERNAME="$(printf '%s' "$USER_JSON" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("username",""))' 2>/dev/null || true)"
if [ -z "$USERNAME" ]; then
  echo "[!] token 无效或无 api 权限。返回："
  printf '%s\n' "$USER_JSON" | head -c 300; echo
  exit 1
fi
NAME="$(printf '%s' "$USER_JSON" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("name",""))' 2>/dev/null)"
echo "   身份: $NAME (@$USERNAME)  命名空间: $BASE/$USERNAME"
[ "${1:-}" = "--check" ] && exit 0

echo "== 2. 创建项目（已存在则跳过）=="
for spec in "${REPOS[@]}"; do
  IFS=: read -r dir repo desc <<<"$spec"
  enc="$(python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "$USERNAME/$repo")"
  code="$(curl -sS -m 30 -o /dev/null -w '%{http_code}' -H "PRIVATE-TOKEN: $TOKEN" "$API/projects/$enc")"
  if [ "$code" = "200" ]; then
    echo "   [$repo] 已存在"
  else
    body="$(python3 -c 'import json,sys;print(json.dumps({"name":sys.argv[1],"path":sys.argv[1],"description":sys.argv[2],"visibility":"private","initialize_with_readme":False}))' "$repo" "$desc")"
    r="$(api POST /projects "$body")"
    ok="$(printf '%s' "$r" | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d.get("http_url_to_repo",""))' 2>/dev/null || true)"
    if [ -n "$ok" ]; then echo "   [$repo] 已创建 → $ok"
    else echo "   [$repo] 创建失败："; printf '%s\n' "$r" | head -c 300; echo; fi
  fi
done
[ "${1:-}" = "--create" ] && exit 0

echo "== 3. 推送 =="
HELPER='!f() { echo "password=$(cat ~/.sourcefind-token)"; }; f'
for spec in "${REPOS[@]}"; do
  IFS=: read -r dir repo desc <<<"$spec"
  url="$BASE/$USERNAME/$repo.git"
  echo "--- $dir → $url"
  (
    cd "$ROOT/$dir"
    git remote remove origin 2>/dev/null || true
    git remote add origin "https://$USERNAME@developer.sourcefind.cn/codes/$USERNAME/$repo.git"
    git -c credential.helper="$HELPER" push -u origin main
  )
done

echo
echo "== 完成 =="
for spec in "${REPOS[@]}"; do
  IFS=: read -r dir repo desc <<<"$spec"
  echo "  $BASE/$USERNAME/$repo"
done
echo
echo "下一步：把上面两个 URL 填进各仓库 README.md 的「提交仓库：【URL 待填】」再推一次。"
