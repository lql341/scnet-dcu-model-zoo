#!/usr/bin/env bash
# OpenBMB/Trident-Qwen3VL-2B on BW1000 + SGLang 0.5.10（T02 混合模态检索嵌入模型）
# config.json: architectures=['Qwen3VLForEmbedding'], model_type=qwen3_vl
# 与 chat 模型的关键差异：必须 --is-embedding，接口是 /v1/embeddings 而非 /v1/chat/completions
export MODEL_ID="OpenBMB/Trident-Qwen3VL-2B"
export MODEL_PATH="/models/Trident"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/Trident-Qwen3VL-2B"
export SERVED_NAME="trident-qwen3vl-2b"

export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

export TP=1
export MAX_LEN=8192
export MEM_FRAC=0.6
export MAX_RUNNING_REQUESTS=64
export LOAD_TIMEOUT=1800
export MULTIMODAL=1
# 嵌入模型：不生成 token，必须开启 is-embedding
export IS_EMBEDDING=1

export EXTRA_ARGS="--attention-backend triton --trust-remote-code --enable-multimodal --is-embedding"
# --attention-backend triton 必需：本镜像 aiter backend 残缺
# --is-embedding 必需：Qwen3VLForEmbedding 不产生 logits，不加会启动失败
