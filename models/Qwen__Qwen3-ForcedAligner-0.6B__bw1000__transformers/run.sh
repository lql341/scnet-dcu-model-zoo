#!/usr/bin/env bash
# 本条目不是 serving 任务：词级对齐是一次性批处理，没有常驻服务。
# 执行入口是 baseline 下的“一次性推理”，请直接跑 container/eval_official.slurm。
set -euo pipefail
source ./env.sh
mkdir -p logs
cat <<'MSG'
[run] 本条目无需常驻服务。
[run] 运行官方 30 条评测：    sbatch container/eval_official.slurm
[run] 产物：outputs/<id>.json（词级时间戳，单位秒）、反馈表.csv、eval_result.json
MSG
