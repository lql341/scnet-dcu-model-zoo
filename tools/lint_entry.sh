#!/usr/bin/env bash
# 按 docs/entry-spec.md 校验条目
#   ./tools/lint_entry.sh                 # 校验全部
#   ./tools/lint_entry.sh models/<dir>    # 校验单个
set -uo pipefail
cd "$(dirname "$0")/.."
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(models/*/)
FAIL=0
err(){ printf '  ✗ %s\n' "$*"; FAIL=1; }
ok(){ printf '  ✓ %s\n' "$*"; }

for d in "${TARGETS[@]}"; do
  d="${d%/}"; [ -d "$d" ] || continue
  echo "[lint] $(basename "$d")"

  # 1 必备文件
  for f in README.md env.sh image.lock env_check.sh run.sh smoke.sh verify.py \
           requirements-extra.txt MISSING.md meta.json container/submit.slurm \
           logs/env.log logs/smoke.log; do
    [ -e "$d/$f" ] || err "缺 $f"
  done
  [ -d "$d/patches" ] || err "缺 patches/"
  ls "$d"/logs/server.log* >/dev/null 2>&1 || err "缺 logs/server.log(.gz)"

  # 2 README 8 章
  for n in 1 2 3 4 5 6 7 8; do
    grep -qE "^## ${n}\. " "$d/README.md" || err "README 缺第 ${n} 章"
  done
  # 【URL 待填】是硬错误；§6 的【待填】合法——只要该章声明了 [BLOCKED]
  grep -q '【URL 待填】' "$d/README.md" && err "README 仍有【URL 待填】"
  if grep -q '【待填】\|【】' "$d/README.md"; then
    grep -q '\[BLOCKED\]' "$d/README.md" || err "README 有占位符但 §6 未声明 [BLOCKED]"
  fi

  # 3 meta.json 合法性 + 与 env.sh 一致
  python3 - "$d" <<'PY' || FAIL=1
import json, os, re, sys
d = sys.argv[1]
try:
    m = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
except Exception as e:
    print(f'  ✗ meta.json 解析失败: {e}'); sys.exit(1)
need = ['model_id','org','model','card','cards_used','tp','framework','framework_version',
        'precision','status','bench','cold_start_s','throughput','optimized','verified_on']
missing = [k for k in need if k not in m]
if missing: print(f'  ✗ meta.json 缺字段 {missing}'); sys.exit(1)
if m['bench'] not in ('yes','no-material','not-required'):
    print(f"  ✗ meta.bench 取值非法: {m['bench']}"); sys.exit(1)
if m['bench'] == 'yes':
    if not os.path.exists(os.path.join(d,'bench','summary.md')):
        print('  ✗ bench=yes 但缺 bench/summary.md'); sys.exit(1)
    for f in ('round1.jsonl','round2.jsonl'):
        if not os.path.exists(os.path.join(d,'bench',f)):
            print(f'  ✗ bench=yes 但缺 bench/{f}'); sys.exit(1)
else:
    if not m.get('bench_note'):
        if not m.get('bench_note'):
            print('  ✗ bench!=yes 必须在 bench_note 写明原因'); sys.exit(1)

# env.sh 的 IMAGE 必须与 image.lock 的 repository:tag 一致
env = open(os.path.join(d,'env.sh'), encoding='utf-8').read()
img = re.search(r'^export IMAGE="([^"]+)"', env, re.M)
lock = open(os.path.join(d,'image.lock'), encoding='utf-8').read()
repo = re.search(r'\| repository \| `([^`]+)` \|', lock)
tag = re.search(r'\| tag \| `([^`]+)` \|', lock)
if not (img and repo and tag):
    print('  ✗ 无法解析 IMAGE / image.lock'); sys.exit(1)
want = f"image.ac.com:5000/{repo.group(1)}:{tag.group(1)}"
if img.group(1) != want:
    print(f'  ✗ image.lock 与 env.sh 不一致\n      env.sh: {img.group(1)}\n      lock  : {want}'); sys.exit(1)
if not re.search(r'\| digest \| `sha256:[0-9a-f]{64}` \|', lock):
    print('  ✗ image.lock 缺 digest'); sys.exit(1)
sys.exit(0)
PY

  # 4 throughput 必须与 bench/summary.md 逐字一致
  if [ -f "$d/bench/summary.md" ]; then
    python3 - "$d" <<'PY' || FAIL=1
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d,'meta.json'), encoding='utf-8'))
hdr=None; real={}
for line in open(os.path.join(d,'bench','summary.md'), encoding='utf-8'):
    if not line.startswith('|'): continue
    c=[x.strip() for x in line.strip().strip('|').split('|')]
    if c[0]=='random_input_len': hdr=c; continue
    if hdr is None or not c[0].isdigit(): continue
    real[f"{int(c[0])//1024}k_c{int(c[2])}"]=round(float(c[hdr.index('output_throughput')]),2)
bad=[(k, m['throughput'].get(k), v) for k,v in real.items() if m['throughput'].get(k)!=v]
if bad:
    for k,a,b in bad: print(f'  ✗ throughput[{k}] meta={a} summary={b}')
    sys.exit(1)
sys.exit(0)
PY
  fi

  # 5 体积
  kb=$(du -sk "$d" | awk '{print $1}')
  [ "$kb" -gt 2048 ] && err "条目 ${kb}KB 超过 2MB"
  big=$(find "$d" -type f ! -name '*.gz' ! -path '*/bench/round*.jsonl' -size +1024k | head -3)
  [ -n "$big" ] && err "单文件超 1MB: $(echo "$big" | tr '\n' ' ')"
  srv=$(ls -l "$d"/logs/server.log 2>/dev/null | awk '{print int($5/1024)}')
  [ -n "${srv:-}" ] && [ "$srv" -gt 100 ] && err "logs/server.log ${srv}KB 未压缩（规范要求 >100KB 压 gzip）"

  [ $FAIL -eq 0 ] && ok "通过" || true
done
echo
[ $FAIL -eq 0 ] && echo "[lint] 全部通过" || echo "[lint] 存在不合规项"
exit $FAIL
