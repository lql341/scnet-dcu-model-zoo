#!/usr/bin/env bash
# 模型、镜像、框架、卡型、并行方式与启动参数；其余脚本读这里的变量。
# 注意：MODEL_PATH 是【容器内】挂载点；宿主机真实路径见 HOST_MODEL_PATH。

# ---------- 模型 ----------
export MODEL_ID="XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
export MODEL_PATH="/models/MiMo"                       # 容器内挂载点
export HOST_MODEL_PATH="/public/home/scnethr2601a10/models/MiMo-V2.6-Distill-Qwen-9B"
export SERVED_NAME="mimo-v2.6-9b"

# ---------- 镜像 / 框架 / 卡型 ----------
export IMAGE="image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1"
export HOST_IMAGE_TAR="/public/home/scnethr2601a10/images/mimo-sglang-bw10000_1.1.1.tar"
export FRAMEWORK="sglang"                              # sglang | vllm
export CARD="BW1000"                                   # BW1000 | K100-AI（K500-SM 按 K100-AI 计）
export PRECISION="bf16"                                # bf16 | int8 | int4
export PORT=30000

# ---------- 并行 ----------
export TP=1                                            # 9B / 单卡 64GB，无需切分
export EP=1
export PP=1
export NNODES=1
export NODE_RANK=0
export DIST_ADDR="10.0.0.1:50000"

# ---------- 显存 / 上下文 ----------
export MAX_LEN=20480                                   # 覆盖 bench 的 16K 输入 + 1K 输出 + chat template 开销（17408 不够，实测会 400）
export MEM_FRAC=0.85
export MAX_RUNNING_REQUESTS=64               # 与 bench 最高并发档对齐；=8 时 4K×64 档只有 176 tok/s（请求排队），改 64 后 998 tok/s
export LOAD_TIMEOUT=1800                               # 实测冷启动 410s（含 Triton JIT）

# ---------- 适配参数 ----------
export PATCH_ROOT="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export EXTRA_ARGS="--attention-backend triton"   # CUDA graph 保持开启（实测比 --disable-cuda-graph 快 73%~466%）

# 海光专属后端开关（来自实测可跑通组合）
export SGLANG_USE_LIGHTOP=1
export SGLANG_KV_LAYOUT_DCU_FA=0
export SGLANG_USE_AITER_FP8_ASM_MOE=1
export SGLANG_USE_TRITON_EXTEND_FROM_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
