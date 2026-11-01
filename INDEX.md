# 实测索引

> 本文件由 `tools/gen_index.py` 从各条目 `meta.json` 生成，**请勿手改**。

## 已完成 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | TP | 冷启动 | 4K/c1 | 4K/c8 | 4K/c64 | 16K/c1 | 16K/c8 | 16K/c64 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | sglang 0.5.10rc0 | bf16 | 1 | 400–410 s | 44.31 | 360.23 | 998.05 | 41.47 | 308.72 | 299.81 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | vllm 0.20.0 | bf16 | 1 | 270–410 s | 20.73 | 160.63 | 489.52 | 8.19 | 44.83 | 100.82 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) |
| Kimi-Linear-48B-A3B-Instruct | BW1000 ×4 | sglang 0.5.10rc0 | bf16 | 4 | 280–360 s | 14.52 | 111.01 | 608.23 | 14.01 | 80.73 | 249.89 | [→](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) |

单位：output tok/s（取 `bench/round2.jsonl`）。

## 已跑通但无 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | 状态 | 冷启动 | 原因 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-ASR-1.7B | BW1000 ×1 | vllm | bf16 | smoke | 150–200 s | T04 语音类性能测法随分题材料下发，尚未获取 | [→](models/Qwen__Qwen3-ASR-1.7B__bw1000__vllm/) |

## 未完成 / 失败记录

跑不通的配置也留档，写明原因，不删。

| 组合 | 结果 | 原因 | 出处 |
| --- | --- | --- | --- |
| Kimi-Linear × BW1000 × vLLM | ❌ 框架不支持 | vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio） | gotchas #2 旁注 |
| Kimi-Linear × BW1000 × SGLang × CUDA Graph | 🟡 未完成 | TP=4 报 `HIP error: invalid device ordinal` | gotchas #14 |
| MiMo × K100-AI × vLLM | 🟡 仅 smoke | Notebook 内跑通，6 档基准未跑 | 见 README 附注 |
| MiMo × K100-AI × SGLang | ❌ 镜像缺陷 | `jupyter-sglang` 镜像 aiter 缺 `ops.triton.gemm` | gotchas #2 |
