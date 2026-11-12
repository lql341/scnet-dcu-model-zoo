#!/usr/bin/env bash
# moonshotai/Kimi-Linear-48B-A3B-Instruct on BW1000 + **vLLM 0.20**
# 与同目录的 __sglang 条目配对：同一模型、同一卡型，换引擎做硬对比。
# 关键事实（2026-11-12 实测）：vLLM 0.20 的 ModelRegistry 含 `KimiLinearForCausalLM`，
# 架构受支持；此前"HANDOFF 记 vLLM 不支持 kimi_linear"的判断需修正。

export MODEL_ID="moonshotai/Kimi-Linear-48B-A3B-Instruct"
export MODEL_PATH="/models/Kimi"
export HOST_MODEL_PATH="/public/home/luql/models/Kimi-Linear-48B-A3B-Instruct"
export SERVED_NAME="kimi-linear-48b"

export IMAGE="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
export CLI_IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"   # 仅压测客户端
export HOST_CLI_TAR=""
export FRAMEWORK="vllm"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# ---------- 并行与显存 ----------
# 92 GB BF16 权重 → TP=4 约 23 GB/卡，余量给 KV cache
export TP=4
export EP=1
export PP=1
export MAX_LEN=20480
export MEM_FRAC=0.92                # vLLM 在 DCU 上探测偏保守，0.85 会报 cache blocks 不足
export MAX_NUM_SEQS=128
export LOAD_TIMEOUT=2400             # Kimi 48B + TP=4 冷启动较慢

export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS"
