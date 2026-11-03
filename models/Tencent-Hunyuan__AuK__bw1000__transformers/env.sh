#!/usr/bin/env bash
# Tencent-Hunyuan/AuK on BW1000 + Transformers（T04 带参考音语音合成）
# 注意：AuK 官方 pin 是 torch>=2.7,<2.8 + transformers>=4.52,<5，
# 而国产卡镜像是 DCU 版 torch 2.9/2.10 + transformers 5.x（PyPI 无 DCU 版 torch 2.7，无法降级）。
# 因此本条目试「忽略 pin、用镜像自带 torch」——能否跑通以实测为准。
export MODEL_ID="Tencent-Hunyuan/AuK"
export MODEL_PATH="/models/AuK"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/AuK"
export ENCODER_PATH="/models/Qwen2.5-Omni-3B"
export HOST_ENCODER_PATH="/public/home/scnethr2601a10/models/Qwen2.5-Omni-3B"
export SERVED_NAME="auk"

# 选 SGLang 镜像而非 vLLM 镜像：AuK 的 infer_cli 需要 torchaudio，
# 该镜像自带 DCU 版 torchaudio 2.9.0+das；vLLM 镜像无 torchaudio，而 PyPI 的 torchaudio
# 是 CUDA/CPU 构建，装进去与 DCU 版 torch 不匹配。
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="transformers"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

export TASK="tts"
export LOAD_TIMEOUT=1800
export WHEELS_HOST="/public/home/scnethr2601a10/wheels/auk-fix"
# 【2026-11-03】原 auk-bin/ 里放的是 cp312 轮子，而镜像是 py3.10，pip 直接拒装 →
# 120 条全失败在 `ModuleNotFoundError: No module named 'librosa'`。
# 现改为按 --python-version 310 --platform manylinux2014_x86_64 重新下载的 cp310 轮子，
# 单一目录（纯 Python + 二进制混放），不再需要 torchcodec（torchaudio.load 已被 soundfile 替掉）。
export WHEELS_BIN_HOST="/public/home/scnethr2601a10/wheels/auk-fix"
export AUK_SRC_HOST="/public/home/scnethr2601a10/AuK-src"
