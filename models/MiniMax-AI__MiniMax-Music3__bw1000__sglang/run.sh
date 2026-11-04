#!/usr/bin/env bash
# 本条目不是 serving 任务（一次性批处理出视频）。入口：sbatch container/submit.slurm
set -euo pipefail
echo "[run] 本条目由 container/inner.sh 先起 harness 服务再评测；入口：sbatch container/submit.slurm"
