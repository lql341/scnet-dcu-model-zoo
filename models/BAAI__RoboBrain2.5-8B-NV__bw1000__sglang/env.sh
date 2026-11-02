#!/usr/bin/env bash
# BAAI/RoboBrain2.5-8B-NV on BW1000 + SGLang 0.5.10（T02 具身大脑 / 视觉多模态）
# 标准 Qwen3VLForConditionalGeneration（qwen3_vl，结构与 Qwen3-VL-8B 相同）：无需装包、无需 patch
export MODEL_ID="BAAI/RoboBrain2.5-8B-NV"
export MODEL_PATH="/models/RoboBrain"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/RoboBrain2.5-8B-NV"
export SERVED_NAME="robobrain2.5-8b-nv"

export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

export TP=1
export MAX_LEN=32768
export MEM_FRAC=0.85
export MAX_RUNNING_REQUESTS=64
export LOAD_TIMEOUT=1800
export MULTIMODAL=1

export EXTRA_ARGS="--attention-backend triton --trust-remote-code --enable-multimodal"
# --attention-backend triton 必需：本镜像的 aiter attention backend 残缺，默认选择会在首个 decode 请求崩溃
