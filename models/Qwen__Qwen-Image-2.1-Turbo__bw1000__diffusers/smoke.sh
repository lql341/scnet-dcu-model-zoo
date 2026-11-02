#!/usr/bin/env bash
# 管线 smoke 已并入 evaluate_official.py 的 PROBE_ONLY 模式
set -euo pipefail
echo "[smoke] 见 container/submit.slurm 中的探针阶段（PROBE_ONLY=3）"
