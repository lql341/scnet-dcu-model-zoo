#!/usr/bin/env bash
# Tencent-Hunyuan/HY-World-2.0（WorldMirror 2.0 重建部分）× BW1000
# 复用 Harbor 上社区作者的现成镜像（内含 gsplat 的 HIP 补丁与完整 harness）

export MODEL_ID="Tencent-Hunyuan/HY-World-2.0"
export MODEL_PATH="/models/HY-World-2.0"
export HOST_MODEL_PATH="/public/home/luql/models/HY-World-2.0"

export IMAGE="image.ac.com:5000/dcu/acizv7iztz/base/hy-world2-bw-sglang:20261103-rc1"
export HOST_IMAGE_TAR=""
export FRAMEWORK="sglang"
export FRAMEWORK_VERSION="vLLM 无；SGLang Diffusion 接入 + 官方 WorldMirror PyTorch 管线"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# harness（镜像作者的适配工程，已归档到 zoo-images/assets/hyworld2-bw-harness.tar.gz）
export HARNESS_HOST="/public/home/luql/work/hyw-extract/harness"
export MATERIALS_HOST="/public/home/luql/work/hyw/eval-materials"

export VIEW_COUNT=10
export LOAD_TIMEOUT=1800
