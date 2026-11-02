# 实测索引

> 本文件由 `tools/gen_index.py` 从各条目 `meta.json` 生成，**请勿手改**。

## 已完成 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | TP | 冷启动 | 4K/c1 | 4K/c8 | 4K/c64 | 16K/c1 | 16K/c8 | 16K/c64 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | sglang 0.5.10rc0 | bf16 | 1 | 400–410 s | 44.31 | 360.23 | 998.05 | 41.47 | 308.72 | 299.81 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | vllm 0.20.0 | bf16 | 1 | 270–410 s | 20.73 | 160.63 | 489.52 | 8.19 | 44.83 | 100.82 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) |
| Kimi-Linear-48B-A3B-Instruct | BW1000 ×4 | sglang 0.5.10rc0 | bf16 | 4 | 280–360 s | 104.67 | 353.53 | 839.00 | 50.80 | 119.21 | 234.17 | [→](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) |

单位：output tok/s（取 `bench/round2.jsonl`）。

## 已跑通但无 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | 状态 | 冷启动 | 原因 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Recon2Reason-Reasoning-4B | BW1000 ×1 | sglang | bf16 | smoke | 150–150 s | T02 多模态评测器与性能测法随分题材料下发，尚未获取；本件完成的是视觉链路 smoke | [→](models/BAAI__Recon2Reason-Reasoning-4B__bw1000__sglang/) |
| RoboBrain2.5-8B-NV | BW1000 ×1 | sglang | bf16 | smoke | 150–200 s | T02 多模态评测器与性能测法随分题材料下发，尚未获取；本件完成视觉链路 smoke | [→](models/BAAI__RoboBrain2.5-8B-NV__bw1000__sglang/) |
| Qwen3-ASR-1.7B | BW1000 ×1 | vllm | bf16 | official-eval | 150–200 s | 官方 T04 明确「不进行统一评分和性能比较」，不要求服务接口；已按官方 300 条条件完成评测 | [→](models/Qwen__Qwen3-ASR-1.7B__bw1000__vllm/) |
| Qwen3-ForcedAligner-0.6B | BW1000 ×1 | transformers | bf16 | official-eval | 0–0 s | 官方 T04 明确不进行统一评分和性能比较；且 BOUNDARIES_PENDING —— 无人工词边界，仅收集预测不计算准确率 | [→](models/Qwen__Qwen3-ForcedAligner-0.6B__bw1000__transformers/) |

## 失败记录（blocked）

跑不通的配置也留档，写明原因与证据，不删。

| 组合 | 结果 | 原因 | 详情 |
| --- | --- | --- | --- |
| Trident-Qwen3VL-2B × BW1000 × sglang | ❌ 服务无法启动 | 框架不支持 Qwen3VLForEmbedding 架构，服务无法启动 | [→](models/OpenBMB__Trident-Qwen3VL-2B__bw1000__sglang/) |
| Trident-Qwen3VL-2B × BW1000 × vllm | ❌ 服务无法启动 | vLLM 0.20 不支持 Qwen3VLForEmbedding 架构，服务无法启动 | [→](models/OpenBMB__Trident-Qwen3VL-2B__bw1000__vllm/) |

| 组合 | 结果 | 原因 |
| --- | --- | --- |
| Kimi-Linear × BW1000 × vLLM | ❌ 框架不支持 | vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio） | gotchas #2 旁注 |
| Kimi-Linear × BW1000 × SGLang × CUDA Graph | 🟡 未完成 | TP=4 报 `HIP error: invalid device ordinal` | gotchas #14 |
| MiMo × K100-AI × vLLM | 🟡 仅 smoke | Notebook 内跑通，6 档基准未跑 | 见 README 附注 |
| MiMo × K100-AI × SGLang | ❌ 镜像缺陷 | `jupyter-sglang` 镜像 aiter 缺 `ops.triton.gemm` | gotchas #2 |
