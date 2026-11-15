#!/usr/bin/env bash
# 把本地 zoo 同步到 luql 的 zoo-sync（作业脚本从这里 rsync 到自己 WORK）。
#
# 为什么需要它：作业的 slurm 脚本在启动时 `rsync zoo-sync/<entry> -> $WORK/repo`。
# 只改本地、忘了推远程，作业就会跑旧脚本。2026-11-15 因此让"有效性校验"带着
# 第二个 NameError（`ILENS` 未定义）跑了一整个下午（gotchas #45）。
#
# 用法：./tools/sync_remote.sh [条目名 ...]      不给参数 = 全量 sweeps/ + models/ 下存在的条目
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

HOST='luql@zzeshell.scnet.cn'
PORT=65032
DEST=/public/home/luql/work/zoo-sync
SSH="ssh -p $PORT -o ConnectTimeout=25"

# 远程实际存在哪些条目：只推这些，其余会以 "unexpected end of file" 失败
REMOTE=$($SSH "$HOST" "ls $DEST/models 2>/dev/null" | tr -d '\r')
[ -z "$REMOTE" ] && { echo "远程 $DEST/models 为空或不可达"; exit 1; }
echo "远程条目：$(echo $REMOTE | wc -w | tr -d ' ') 个"

want="$*"
[ -z "$want" ] && want="$REMOTE"

for e in $want; do
  if [ ! -d "models/$e" ]; then echo "  skip $e（本地不存在）"; continue; fi
  if ! echo "$REMOTE" | grep -qx "$e"; then echo "  skip $e（远程不存在）"; continue; fi
  rsync -a --exclude bench --exclude logs --exclude official-eval \
    "models/$e/" "$HOST:$DEST/models/$e/" || { echo "  FAIL $e"; exit 1; }
  echo "  ok $e"
done

rsync -a sweeps/ "$HOST:$DEST/sweeps/" 2>/dev/null && echo "  ok sweeps/"

# 推完自检：坏行一处都不许留
bad=$($SSH "$HOST" "grep -rln 'EXPECT = len(ILENS' $DEST/models/ 2>/dev/null | wc -l" | tr -d '\r')
echo "远程 bench.sh 残留坏行文件数：$bad"
[ "$bad" = "0" ] || { echo "仍有坏副本，检查上面的路径"; exit 1; }
