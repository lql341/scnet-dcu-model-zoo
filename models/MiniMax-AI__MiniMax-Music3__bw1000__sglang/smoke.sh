#!/usr/bin/env bash
# 管线 smoke 已并入 evaluate_official.py：PROBE_ONLY=1 只跑第 0 片的第 1 条
set -euo pipefail
echo "[smoke] 见 container/submit.slurm：PROBE_ONLY=2 sbatch container/submit.slurm"
