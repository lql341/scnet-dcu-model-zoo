#!/usr/bin/env bash
# MiniMax-Music3 on BW1000 + 镜像自带的 Hygon SGLang 运行时（T03 音乐生成）
# 注意：Music3 不是常规 LLM serving，必须用 Harbor 上这个 BW 专用镜像，
# 里面带了 SGLang-Omni 的 RVQ / flow-matching / DAV 组件与 Hygon 适配。

export MODEL_ID="MiniMaxAI/MiniMax-Music3"
export MODEL_PATH="/models/Music3"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/MiniMax-Music3"
export SERVED_NAME="MiniMaxAI/MiniMax-Music3"

# 镜像：BW 专用（41 层，压缩 12.27 GB），Harbor 上匿名可 pull
export IMAGE="image.ac.com:5000/dcu/acizv7iztz/base/minimax-music3-bw-sglang:20261031-rc1"
export HOST_IMAGE_TAR=""     # 上游无本地 tar；首次直接 pull（~3 min）
export FRAMEWORK="sglang"
export FRAMEWORK_VERSION="0.5.12（镜像内 Hygon 构建）"
export CARD="BW1000"
export PRECISION="bf16"
export ACOUSTIC_PRECISION="fp32"   # 声学（flow-matching/DAV）走 fp32，与上游适配一致
export PORT=8000

# 官方条件（60 条：中英各 10 歌词 × 3 seed）
export CONDITIONS_HOST="/public/home/scnethr2601a10/work/official-conds/music3.jsonl"
export MAX_NEW_TOKENS=750          # 条件清单 params.max_new_tokens（≤9000 帧上限）
export MEM_FRAC=0.5
export MAX_LEN=10240
export LOAD_TIMEOUT=1800

# BW 适配 harness（来自镜像 /tmp/music3-hygon，已归档到 zoo-images/assets/music3-hygon-harness.tar.gz）
# 它负责：打兼容 patch、建 venv、起 OpenAI 兼容服务（/v1/audio/speech）
export HARNESS_HOST="/public/home/scnethr2601a10/work/music3-hygon"
