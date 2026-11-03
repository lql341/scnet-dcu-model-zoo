#!/usr/bin/env bash
# Wan2.1-I2V-14B-480P on BW1000（官方 Wan2.1 代码，非 diffusers 管线）
# 采集环境 → logs/env.log；其余脚本读这里的变量。
# 注意：MODEL_PATH 是【容器内】挂载点；宿主机真实路径见 HOST_MODEL_PATH。

# ---------- 模型 ----------
export MODEL_ID="Wan-AI/Wan2.1-I2V-14B-480P"
export MODEL_PATH="/models/Wan21"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/Wan2.1-I2V-14B-480P"
export TASK="i2v-14B"

# ---------- 镜像 / 框架 / 卡型 ----------
# 用 SGLang 镜像：DTK 26.04 + DCU 版 torch 2.9（Wan2.1 官方代码的 flash_attn 会缺失，
# wan/modules/attention.py 会自动回落到 torch.nn.functional.scaled_dot_product_attention）
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="wan2.1"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# ---------- 官方代码与依赖 ----------
export WAN_SRC_HOST="/public/home/scnethr2601a10/Wan2.1-src"
# diffusers 复用 Qwen-Image 条目归档的那份（hub 1.33.0 与之配套），纯 Python 依赖单独一套
export WHEELS_HOST="/public/home/scnethr2601a10/wheels/wan-fix"
export WHEELS_QI_HOST="/public/home/scnethr2601a10/wheels/qi2"

# ---------- 并行 ----------
# 官方 60 条条件，按卡分片跑（1 卡 1 进程）。SHARD_TOTAL 由 submit.slurm 按申请卡数设置。
export SHARD_TOTAL="${SHARD_TOTAL:-1}"
export OFFLOAD_MODEL="${OFFLOAD_MODEL:-True}"
# 单卡 64 GB 装不下 14B + UMT5-XXL（实测 OOM），把 T5 放 CPU
export T5_CPU="${T5_CPU:-1}"

# ---------- 官方固定条件（取自 conditions/wan_i2v.jsonl 的 params） ----------
export WIDTH=832
export HEIGHT=480
export FRAME_NUM=81
export FPS=16
export SAMPLE_STEPS=40
export SAMPLE_GUIDE_SCALE=5.0
export SAMPLE_SHIFT=3.0
export SAMPLE_SOLVER="unipc"
export LOAD_TIMEOUT=1800
