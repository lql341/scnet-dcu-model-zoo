#!/usr/bin/env bash
# 无常驻服务：一次性批处理（5 条 DTU 场景重建）
set -euo pipefail
echo "[run] 请用: sbatch container/submit.slurm"
echo "[run] 或容器内: bash build_gsplat.sh && python3 run_conditions.py --ids dtu_scan1 ..."
