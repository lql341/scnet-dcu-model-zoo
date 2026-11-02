#!/usr/bin/env bash
# 本条目不是 serving 任务，无常驻服务。入口：sbatch container/submit.slurm
set -euo pipefail
echo "[run] 一次性批处理任务，请用: sbatch container/submit.slurm"
