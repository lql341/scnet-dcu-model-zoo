#!/usr/bin/env bash
# Qwen/Qwen3-VL-8B-Instruct on BW1000 + vLLM 0.21（T02 视觉多模态理解）
# 架构 Qwen3VLForConditionalGeneration，与 BAAI/RoboBrain2.5-8B-NV 同底座：无需装包、无需 patch。
# 选 0.21 的原因见 image.lock：0.20 的 deepstack 断言会让任何带图请求 500。
export MODEL_ID="Qwen/Qwen3-VL-8B-Instruct"
export MODEL_PATH="/models/M"
export HOST_MODEL_PATH="/public/home/luql/models/Qwen3-VL-8B-Instruct"
export SERVED_NAME="qwen3-vl-8b-instruct"

export IMAGE="image.ac.com:5000/dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1"
# ⚠️ 2026-11-22：Harbor 在本集群已不可达（connect timeout / no route to host），
# 且该仓库曾报 artifact not found。**必须**走本地归档：这份 tar 是未压缩态（31 GB），
# 里面 manifest.json 的 RepoTags 与上面 $IMAGE 完全一致，podman load 后 tag 自动对上。
export HOST_IMAGE_TAR="/public/home/luql/work/imgstore/vllm021.tar"
export FRAMEWORK="vllm"
export FRAMEWORK_VERSION="0.21.0（vllm0.21.0-ubuntu22.04-dtk2604-py3.10 镜像）"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# 权重 16.34 GiB（4 个 safetensors），bf16 单卡 BW1000（64 GB）放得下且余量充足
export TP=1
export MAX_LEN=32768
export MEM_FRAC=0.95
export MAX_NUM_SEQS=128
export LOAD_TIMEOUT=1800
export MULTIMODAL=1

# ---------- vLLM 专用 ----------
export CLI_IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"   # 仅压测客户端
# 同样走本地归档（7.9 GB gzip），脚本里会自动 gunzip 流式 load
export HOST_CLI_TAR="/public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz"
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS "
