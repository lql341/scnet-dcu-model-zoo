#!/usr/bin/env bash
# 仓库体积守门：防止跑多了模型后仓库膨胀
#   - 单条目 ≤ 2 MB
#   - 全库 ≤ 25 MB
#   - 任何文件 ≤ 1 MB（除 bench/*.jsonl 明文数据）
set -uo pipefail
cd "$(dirname "$0")/.."

FAIL=0
MAX_ENTRY=$((2*1024))       # KB
MAX_TOTAL=$((25*1024))
MAX_FILE=$((1024))

total=$(du -sk . 2>/dev/null | awk '{print $1}')
echo "[quota] 全库 $((total/1024)) MB（上限 $((MAX_TOTAL/1024)) MB）"
[ "$total" -gt "$MAX_TOTAL" ] && { echo "  ✗ 全库超限"; FAIL=1; }

for d in models/*/; do
  sz=$(du -sk "$d" | awk '{print $1}')
  printf "[quota] %-62s %4s KB\n" "$(basename "$d")" "$sz"
  [ "$sz" -gt "$MAX_ENTRY" ] && { echo "  ✗ 单条目超过 $((MAX_ENTRY/1024)) MB"; FAIL=1; }
done

while IFS= read -r f; do
  case "$f" in *bench/round*.jsonl) continue;; esac
  kb=$(du -k "$f" | awk '{print $1}')
  if [ "$kb" -gt "$MAX_FILE" ]; then echo "  ✗ $f = ${kb} KB（单文件上限 $((MAX_FILE/1024)) MB）"; FAIL=1; fi
done < <(find models -type f ! -name '*.gz')

[ $FAIL -eq 0 ] && echo "[quota] OK" || echo "[quota] 有超限项"
exit $FAIL
