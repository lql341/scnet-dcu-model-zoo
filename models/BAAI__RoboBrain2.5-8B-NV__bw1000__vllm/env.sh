#!/usr/bin/env bash
# BAAI/RoboBrain2.5-8B-NV on BW1000 + vLLM 0.20（T02 视觉多模态；与 __sglang 条目配对做跨引擎对比）
# 标准 Qwen3VLForConditionalGeneration（qwen3_vl，结构与 Qwen3-VL-8B 相同）：无需装包、无需 patch
export MODEL_ID="BAAI/RoboBrain2.5-8B-NV"
export MODEL_PATH="/models/M"
export HOST_MODEL_PATH="/public/home/luql/models/RoboBrain2.5-8B-NV"
export SERVED_NAME="robobrain2.5-8b-nv"

export IMAGE="image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1"
export HOST_IMAGE_TAR=""   # vLLM 无本地 tar，直接 pull
export FRAMEWORK="vllm"
export FRAMEWORK_VERSION="0.21.0（vllm0.21.0-ubuntu22.04-dtk2604-py3.10 镜像）"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

export TP=1
export MAX_LEN=32768
export MEM_FRAC=0.95
export MAX_NUM_SEQS=128
export LOAD_TIMEOUT=1800
export MULTIMODAL=1

export EXTRA_ARGS="--attention-backend triton --trust-remote-code --enable-multimodal"
# --attention-backend triton 必需：本镜像的 aiter attention backend 残缺，默认选择会在首个 decode 请求崩溃
# ---------- vLLM 专用 ----------
export CLI_IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"   # 仅压测客户端
export HOST_CLI_TAR=""
# 引擎版本说明：vLLM 0.20 的 qwen3_vl.py 里 _get_deepstack_input_embeds 会硬性报错
#   ValueError: Requested more deepstack tokens than available in buffer: 72 > 67
# vLLM 0.21 已删除该断言（只做切片），因此本条目用 0.21 镜像。
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS "
# 架构：vLLM 0.20 的 ModelRegistry 含 Qwen2_5_VLForConditionalGeneration（2026-11-12 实测）
