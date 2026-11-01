#!/usr/bin/env bash
# 模型、镜像、框架、卡型、并行方式与启动参数；其余脚本读这里的变量。
# 注意：MODEL_PATH 是【容器内】挂载点；宿主机真实路径见 HOST_MODEL_PATH。

# ---------- 模型 ----------
export MODEL_ID="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
export MODEL_PATH="/models/MiMo"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B"
export SERVED_NAME="mimo-v2.6-9b"

# ---------- 镜像 / 框架 / 卡型 ----------
export IMAGE="image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0"
export CLI_IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"   # 仅用于压测客户端
export HOST_CLI_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export HOST_IMAGE_TAR=""   # 该镜像本地无缓存，留空则直接 pull
export FRAMEWORK="vllm"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# ---------- 并行 ----------
export TP=1
export EP=1
export PP=1

# ---------- 显存 / 上下文 ----------
export MAX_LEN=20480                # 覆盖 16K 输入 + 1K 输出 + chat template 开销（17408 不够，实测会 400）
export MEM_FRAC=0.95                # DCU 上 vLLM 探测偏保守，0.85 会报 "No available memory for the cache blocks"
export MAX_NUM_SEQS=128                 # 与 bench 最高并发档对齐（vLLM 用 --max-num-seqs，非 max-running-requests）
export LOAD_TIMEOUT=1800            # 实测冷启动 270–300 s

# ---------- 适配参数 ----------
export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export EXTRA_ARGS="--trust-remote-code --dtype bfloat16 --max-num-seqs $MAX_NUM_SEQS"   # CUDA Graph 保持开启（实测比 --enforce-eager 快 3%~32%）
