# 实测索引

> 本文件由 `tools/gen_index.py` 从各条目 `meta.json` 生成，**请勿手改**。

## 已完成 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | TP | 冷启动 | 4K/c1 | 4K/c8 | 4K/c64 | 16K/c1 | 16K/c8 | 16K/c64 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Recon2Reason-Reasoning-4B | BW1000 ×1 | sglang 0.5.10rc0 | bf16 | 1 | 200–260 s | 78.35 | 441.43 | 719.64 | 57.86 | 226.27 | 149.34 | [→](models/BAAI__Recon2Reason-Reasoning-4B__bw1000__sglang/) |
| Recon2Reason-Reasoning-4B | BW1000 ×1 | vllm 0.20.0（vllm0.20-qwen_asr1.7_bw 镜像） | bf16 | 1 | 200–260 s | 98.83 | 544.43 | 845.30 | 81.93 | 267.09 | 190.66 | [→](models/BAAI__Recon2Reason-Reasoning-4B__bw1000__vllm/) |
| RoboBrain2.5-8B-NV | BW1000 ×1 | sglang 0.5.10rc0 | bf16 | 1 | 160–160 s | 45.69 | 327.14 | 543.80 | 37.14 | 126.65 | 141.66 | [→](models/BAAI__RoboBrain2.5-8B-NV__bw1000__sglang/) |
| RoboBrain2.5-8B-NV | BW1000 ×1 | vllm 0.20.0（vllm0.20-qwen_asr1.7_bw 镜像） | bf16 | 1 | 200–260 s | 64.01 | 387.93 | 667.69 | 51.14 | 225.76 | 175.71 | [→](models/BAAI__RoboBrain2.5-8B-NV__bw1000__vllm/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | sglang 0.5.10rc0 | bf16 | 1 | 400–410 s | 44.31 | 360.23 | 998.05 | 41.47 | 308.72 | 299.81 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang/) |
| MiMo-V2.6-Distill-Qwen-9B | BW1000 ×1 | vllm 0.20.0 | bf16 | 1 | 270–410 s | 20.73 | 160.63 | 489.52 | 8.19 | 44.83 | 100.82 | [→](models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/) |
| Kimi-Linear-48B-A3B-Instruct | BW1000 ×4 | sglang 0.5.10rc0 | bf16 | 4 | 280–360 s | 104.67 | 353.53 | 839.00 | 50.80 | 119.21 | 234.17 | [→](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/) |
| Kimi-Linear-48B-A3B-Instruct | BW1000 ×4 | vllm 0.20.0（vllm0.20-qwen_asr1.7_bw 镜像） | bf16 | 4 | 640–700 s | 40.73 | 304.18 | 858.08 | 37.36 | 197.83 | — | [→](models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__vllm/) |

单位：output tok/s（取 `bench/round2.jsonl`）。

## 已跑通但无 6 档基准

| 模型 | 卡型×卡数 | 框架 | 精度 | 状态 | 冷启动 | 原因 | 详情 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MiniMax-Music3 | BW1000 ×1 | sglang | bf16 | official-eval | 600–700 s | 官方 T03 明确「不进行统一评分和性能比较」；按官方 60 条条件生成 32 kHz 立体声、≤750 声学帧 | [→](models/MiniMax-AI__MiniMax-Music3__bw1000__sglang/) |
| Qwen-Image-2.1-Turbo | BW1000 ×1 | diffusers | bf16 | official-eval | 60–90 s | 官方 T03 明确「不进行统一评分和性能比较」；已按官方 180 条条件完成（2048×2048、8 步、KV cache） | [→](models/Qwen__Qwen-Image-2.1-Turbo__bw1000__diffusers/) |
| Qwen3-ASR-1.7B | BW1000 ×1 | vllm | bf16 | official-eval | 150–200 s | 官方 T04 明确「不进行统一评分和性能比较」，不要求服务接口；已按官方 300 条条件完成评测 | [→](models/Qwen__Qwen3-ASR-1.7B__bw1000__vllm/) |
| Qwen3-ForcedAligner-0.6B | BW1000 ×1 | transformers | bf16 | official-eval | 0–0 s | 官方 T04 明确不进行统一评分和性能比较；且 BOUNDARIES_PENDING —— 无人工词边界，仅收集预测不计算准确率 | [→](models/Qwen__Qwen3-ForcedAligner-0.6B__bw1000__transformers/) |
| AuK-Flash | BW1000 ×1 | transformers | bf16 | official-eval | 47–47 s | 官方 T03 明确「不进行统一评分和性能比较」；AuK-Flash 为 4 步蒸馏版，条件清单 nfe=4 / cfg=0 | [→](models/Tencent-Hunyuan__AuK-Flash__bw1000__transformers/) |
| AuK | BW1000 ×1 | transformers | bf16 | official-eval | 47–47 s | 官方 T03 明确「不进行统一评分和性能比较」；已按官方 120 条条件完成评测（参考音克隆） | [→](models/Tencent-Hunyuan__AuK__bw1000__transformers/) |
| HY-OmniWeaving | BW1000 ×1 | hyvideo | bf16 | official-eval | 600–700 s | 官方 T03 明确「不进行统一评分和性能比较」；已按官方 60 条条件完成（832×480 / 81 帧 / 16 fps） | [→](models/Tencent-Hunyuan__HY-OmniWeaving__bw1000__hyvideo/) |
| HY-World-2.0 | BW1000 ×1 | sglang | bf16 | official-eval | 60–90 s | 官方 T03 明确「不进行统一评分和性能比较」；5 条 DTU 场景重建全部完成（scene/cameras/render 三类产物齐全） | [→](models/Tencent-Hunyuan__HY-World-2.0__bw1000__sglang/) |
| Wan2.1-I2V-14B-480P | BW1000 ×4 | wan2.1 | bf16 | official-eval | 360–420 s | 官方 T03 明确「不进行统一评分和性能比较」；按官方 60 条条件产出 832×480/81 帧/16 fps 视频 | [→](models/Wan-AI__Wan2.1-I2V-14B-480P__bw1000__wan2.1/) |

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
