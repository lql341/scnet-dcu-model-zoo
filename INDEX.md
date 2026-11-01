# 实测索引

## 已完成（含原始日志与 6 档基准）

| 模型 | 卡型×卡数 | 框架 | 精度 | TP | 冷启动 | 4K/c1 | 4K/c64 | 16K/c8 | 16K/c64 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kimi-Linear-48B-A3B | BW1000 ×4 | SGLang 0.5.10 | bf16 | 4 | 280–360 s | 14.52 | 608.23 | 80.73 | 249.89 | [→](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | SGLang 0.5.10 | bf16 | 1 | 400–410 s | 44.31 | 998.05 | 308.72 | 299.81 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | vLLM 0.20.0 | bf16 | 1 | 270–410 s | 20.73 | 489.52 | 44.83 | 100.82 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) |

单位：output tok/s。

## 部分验证

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| MiMo × K100-AI × vLLM | 🟡 smoke 通过，bench 未跑 | Notebook 内 `K500SM_AI` 64 GB，vLLM 0.18.1，240 s 起服务，单/多轮/流式全过 |
| MiMo × K100-AI × SGLang | ❌ 镜像问题 | `jupyter-sglang` 镜像的 aiter 缺 `ops.triton.gemm`，sglang import quark 即崩 |
| Kimi × BW1000 × vLLM | ❌ 框架不支持 | vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio） |
| Kimi × BW1000 × SGLang × CUDA Graph | 🟡 未完成 | TP=4 报 `HIP error: invalid device ordinal`，见 gotchas #14 |

## 待测（按可行性排序）

| 模型 | 题 | 权重 | 卡型 | 备注 |
| --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B | T04 | 4.7 GB | BW1000 | vLLM day-0；int8 全空位 |
| Qwen3-ForcedAligner-0.6B | T04 | 1.8 GB | BW1000 | 最小；时间戳 AAS 指标 |
| AuK | T04 | 3.4 GB | BW1000 | 需另下 Qwen2.5-Omni-3B 编码器 |
| Recon2Reason-Reasoning-4B | T02 | 8.9 GB | BW1000 | 标准 Qwen3-VL，无自定义代码 |
| Trident-Qwen3VL-2B | T02 | 4.3 GB | BW1000 | 嵌入模型，检索指标 |
| RoboBrain2.5-8B-NV | T02 | 17.5 GB | BW1000 | Qwen3-VL 架构 |
| Kimi-VL-A3B-Instruct | T02 | 33 GB | BW1000 | SGLang VLM |
| Qwen-Image-2.1-Turbo | T03 | 32.5 GB | BW1000 | Diffusers pipeline |
| MiniCPM-V-4.7-35B-A3B | T02 | 70 GB | BW1000 | MoE VLM |
| Wan2.1-I2V-14B-480P | T03 | 82 GB | BW1000 | 8 卡 FSDP+USP |
| MiniMax-Music3 | T03 | 57 GB | BW1000 | sglang-omni |
| HY-OmniWeaving | T03 | 55 GB | BW1000 | 8 卡 torchrun |
| MobileVLA-R1 | T02 | 26.9 GB | BW1000 | 无 config.json，26 分卷 |
| HY-World-2.0 | T03 | 174 GB | BW1000 | 4 阶段 pipeline |
| DeepSeek-V4.1-Flash-1017 | T02 | ~1 TB | — | 需多节点，空间不足 |
| DeepSeek-V4-Flash-Vision-Exp | T02 | 609 GB (BF16) | — | 官方仅 FP8/FP4，需先转换 |
