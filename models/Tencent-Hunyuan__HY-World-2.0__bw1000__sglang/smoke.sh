#!/usr/bin/env bash
# smoke 由 harness 的 run_conditions.py 单条模式承担
set -euo pipefail
echo "[smoke] 容器内: python3 run_conditions.py --ids dtu_scan1 --output /work/out-smoke"
