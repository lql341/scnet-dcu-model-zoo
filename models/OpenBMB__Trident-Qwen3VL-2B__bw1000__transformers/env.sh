#!/usr/bin/env bash
# Trident-Qwen3VL-2B on BW1000 —— 自建 OpenAI 兼容 embedding 服务
# 这是本模型唯一可行的第三方运行时：vLLM / SGLang 都没注册 Qwen3VLForEmbedding（见两个 blocked 条目）。
# 注意：MODEL_PATH 是【容器内】挂载点；宿主机真实路径见 HOST_MODEL_PATH。

# ---------- 模型 ----------
export MODEL_ID="OpenBMB/Trident-Qwen3VL-2B"
export MODEL_PATH="/models/Trident"
export HOST_MODEL_PATH="/public/home/luql/models/Trident-Qwen3VL-2B"

# ---------- 镜像 / 卡型 ----------
# 复用 SGLang 镜像：DTK 26.04 + DCU 版 torch 2.9；服务本身只依赖 torch + transformers，
# 不需要 SGLang/vLLM 的推理后端。
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz"
export FRAMEWORK="transformers"
export FRAMEWORK_VERSION="transformers 4.57.1 + 自建 HTTP 服务"
export CARD="BW1000"
export PRECISION="bf16"
export PORT="${PORT:-30000}"

# ---------- 依赖（隔离装到 /opt/trident-libs，不动镜像自带的包） ----------
# transformers 4.57.1 才认识 qwen3_vl；qwen_vl_utils 用于图像 resize 与耗时统计
export WHEELS_HOST="/public/home/luql/wheels/omni-fix"
# qwen_vl_utils 必须 0.0.14：官方文件明确写 `pip install qwen-vl-utils==0.0.14`，
# 0.0.8 的预处理与 Qwen3-VL 的 grid/patch 约定不一致，图像输入会报
# `shape ... is invalid for input of size`（只影响图文，纯文本不受影响）。
export WHEELS_EXTRA_HOST="/public/home/luql/wheels/trident"
export PYLIBS="/opt/trident-libs"

# ---------- 并行 ----------
# embedding 服务按需 batch；单卡即可（2B/bf16 ≈ 4 GB 权重）
export TP="${TP:-1}"
export EMB_BATCH="${EMB_BATCH:-16}"
export EMB_MAX_TOKENS="${EMB_MAX_TOKENS:-4096}"
