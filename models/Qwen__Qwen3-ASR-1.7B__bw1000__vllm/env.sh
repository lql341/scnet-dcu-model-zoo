#!/usr/bin/env bash
# Qwen3-ASR-1.7B on BW1000 + vLLM 0.20.0（语音识别，T04）
export MODEL_ID="Qwen/Qwen3-ASR-1.7B"
export MODEL_PATH="/models/ASR"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/Qwen3-ASR-1.7B"
export SERVED_NAME="qwen3-asr-1.7b"

export IMAGE="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
export HOST_IMAGE_TAR=""
export FRAMEWORK="vllm"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

export TP=1
export MEM_FRAC=0.85
export MAX_LEN=8192
export MAX_NUM_SEQS=128               # 60 条子集实测：16→128 让耗时 30s→26s（−13%），CER/WER 零变化；
                                      # 64 之后开始饱和，瓶颈在音频解码+编码器前向，不在解码调度
export WORKERS=32                     # 客户端并发；与上面的 max-num-seqs 配套（见 README §7.2）
export LOAD_TIMEOUT=1800
export AUDIO="/work/repo/assets/asr_test.wav"
export AUDIO_TEXT="甚至出现交易几乎停滞的情况"

export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS"
