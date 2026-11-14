#!/usr/bin/env bash
# Tencent-Hunyuan/HY-OmniWeaving on BW1000（官方 hyvideo 代码 + 序列并行）
# T03 文生视频：832×480 / 81 帧 / 16 fps

export MODEL_ID="Tencent-Hunyuan/HY-OmniWeaving"
export MODEL_PATH="/models/Omni"
export HOST_MODEL_PATH="/public/home/luql/models/HY-OmniWeaving"

# 镜像：与 Wan2.1 同款 SGLang 镜像（DCU 版 torch 2.9 + torchvision）
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz"
export FRAMEWORK="hyvideo"
export FRAMEWORK_VERSION="官方代码（2026-11 clone）+ torch 2.9.0+das"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# 依赖：官方 requirements 的 self-consistent 组合（transformers 4.57.1 / diffusers 0.32.2 / hub 0.34）
export WHEELS_HOST="/public/home/luql/wheels/omni-fix"
export OMNI_SRC_HOST="/public/home/luql/OmniWeaving-src"

# 并行：generate.py 用 WORLD_SIZE 做序列并行（initialize_parallel_state(sp=WORLD_SIZE)）
export NPROC="${NPROC:-1}"

# 官方固定条件（取自 conditions/omniweaving.jsonl 的 params）
export WIDTH=832
export HEIGHT=480
export FRAME_NUM=81
export FPS=16
export SAMPLE_STEPS=50
export GUIDANCE_SCALE=6.0
export FLOW_SHIFT=7.0
export THINK=false
export LOAD_TIMEOUT=1800
