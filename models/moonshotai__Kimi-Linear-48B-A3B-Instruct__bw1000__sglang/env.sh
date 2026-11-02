#!/usr/bin/env bash
# Kimi-Linear-48B-A3B-Instruct on BW1000 + SGLang 0.5.10
# MODEL_PATH 是【容器内】挂载点；宿主机真实路径见 HOST_MODEL_PATH。

# ---------- 模型 ----------
export MODEL_ID="moonshotai/Kimi-Linear-48B-A3B-Instruct"
export MODEL_PATH="/models/Kimi"
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct"
export SERVED_NAME="kimi-linear-48b"

# ---------- 镜像 / 框架 / 卡型 ----------
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="sglang"
export CARD="BW1000"
export PRECISION="bf16"
export PORT=30000

# ---------- 并行 ----------
# 91.5 GB BF16 权重：TP=4 → 约 22.9 GB/卡，余 31 GB/卡给 KV cache
export TP=4
export EP=1
export PP=1
export NNODES=1
export NODE_RANK=0
export DIST_ADDR="10.0.0.1:50000"

# ---------- 显存 / 上下文 ----------
export MAX_LEN=20480                # 覆盖 16K 输入 + 1K 输出 + chat template 开销（17408 不够，实测会 400）
export MEM_FRAC=0.85                # 实测 4 卡 VRAM 占用 86%
export MAX_RUNNING_REQUESTS=64      # 与 bench 的 64 并发档对齐，避免请求排队导致数字失真
export LOAD_TIMEOUT=1800            # 实测冷启动 360 s（TP=4）

# ---------- 适配参数（本件无 patch、无包升级）----------
export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export EXTRA_ARGS="--attention-backend triton"
# CUDA Graph 保持开启：实测 6 档中 5 档大幅优于 --disable-cuda-graph（见 README §7.1），
# 仅 16K/c64 档回退 6.3%（长上下文 + 最高并发下 graph replay 显存开销）

export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_ROCM_FUSED_DECODE_MLA=0
