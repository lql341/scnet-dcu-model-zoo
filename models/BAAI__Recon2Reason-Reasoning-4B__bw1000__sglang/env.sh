#!/usr/bin/env bash
# BAAI/Recon2Reason-Reasoning-4B on BW1000 + SGLang 0.5.10（T02 视觉多模态理解）
# 标准 Qwen3VLForConditionalGeneration：无自定义代码，无需装包、无需 patch
export MODEL_ID="BAAI/Recon2Reason-Reasoning-4B"
export MODEL_PATH="/models/Recon"
export HOST_MODEL_PATH="/public/home/luql/models/Recon2Reason-Reasoning-4B"
export SERVED_NAME="recon2reason-4b"

export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz"
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
# 必须显式 --attention-backend triton：本镜像的 aiter attention backend 不完整
# （aiter_backend.py 里 paged_attention_ragged 未定义），默认选择会启动后立即崩
