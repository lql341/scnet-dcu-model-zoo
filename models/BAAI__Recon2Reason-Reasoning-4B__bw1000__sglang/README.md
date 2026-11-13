# Recon2Reason-Reasoning-4B on BW1000 × **SGLang 0.5.10** —— ✅ 6 档完成（与 vLLM 配对）

> T02 视觉多模态理解。本条目验证「图像预处理 → vision encoder → LLM」全链路，含**多图**输入。

## 1. 简介

`BAAI/Recon2Reason-Reasoning-4B` 在 `异构加速卡BW（BW1000）` + `SGLang 0.5.10`（bf16）上的适配验证。

智源（BAAI）的空间推理视觉语言模型：由 Qwen3-VL-4B-Instruct 微调而来，强化对度量距离、相对位置、
物体布局与空间关系的推理，支持单图与多图输入。`config.json` 实测：
`architectures=['Qwen3VLForConditionalGeneration']`、`model_type=qwen3_vl`、`torch_dtype=bfloat16`，
BF16 权重 2 分片约 8.3 GB，Apache-2.0。

**与官方部署方式的差异**：官方 README 用 Transformers 直接加载（`transformers==4.57.1`、`torch>=2.6`）。
本件改用 SGLang 提供 OpenAI 兼容服务，唯一必要的适配是**显式指定 `--attention-backend triton`** ——
本镜像默认会选中 `aiter` attention backend，而该 backend 残缺，启动后立即崩（见 §8、`docs/gotchas.md`）。

模型是标准 `Qwen3VLForConditionalGeneration`，**无需 `trust_remote_code` 的自定义代码、无需装包、无需 patch**。

- 权重：`https://www.modelscope.cn/models/BAAI/Recon2Reason-Reasoning-4B`
- 镜像：见 `image.lock`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/recon2reason-4b-bw1000-sglang

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（digest 见 `image.lock`）

```text
torch: 2.9.0+das.opt1.dtk2604.2605140906.gcbed15
HIP: 6.3.26113
sglang: 0.5.10rc0+das.opt2.alpha.dtk2604
triton: 3.5.1+das.opt1.dtk2604.torch290
transformers: 5.3.0
Device: BW × 1，vram 64.0 GB，arch gfx936:sramecc+:xnack-
```

- 卡型 / 卡数：BW1000 × 1（`TP=1`）
- 冷启动：**150 s**
- 模型路径（容器内）：`/models/Recon`
- `/v1/models` 返回 `max_model_len: 32768`

## 3. 适配优化方案

**无软件包升级、无代码 patch。** 参数：

| 参数 | 取值 | 原因 |
| --- | --- | --- |
| `--attention-backend` | `triton` | **必需**。默认会选残包的 `aiter` backend，报 `NameError: name 'paged_attention_ragged' is not defined` |
| `--enable-multimodal` | — | T02 要求图像输入 |
| `--context-length` | `32768` | 模型支持更长，8.3 GB 权重下单卡 64 GB 有余量 |
| `--mem-fraction-static` | `0.85` | |
| `--max-running-requests` | `64` | 对齐压测最高并发档（见 `docs/gotchas.md` #3） |
| CUDA Graph | 保持开启 | 未传 `--disable-cuda-graph` |

容器运行时三条硬要求（`/dev/mkfd`、挂宿主 `/opt/hyhal`、不覆盖镜像内 `/opt/dtk`）见 `docs/gotchas.md`。

## 4. 服务启动脚本

```bash
bash run.sh
```

实际执行：

```bash
python3 -m sglang.launch_server \
  --model-path /models/Recon --served-model-name recon2reason-4b \
  --tp-size 1 --context-length 32768 --mem-fraction-static 0.85 \
  --max-running-requests 64 \
  --host 127.0.0.1 --port 30000 \
  --attention-backend triton --trust-remote-code --enable-multimodal
```

## 5. Smoke 测试

`smoke.sh` 覆盖四项：健康检查、纯文本、**单图**、**多图（2 张）**。
测试图由脚本内纯 Python 即时生成（320×240 纯色 PNG），正确答案唯一可判定。

### 实测结果

| 项 | 结果 |
| --- | --- |
| `/health` | `HTTP 200` |
| `/v1/models` | `recon2reason-4b`，`max_model_len: 32768` |
| 纯文本 | `空间推理是指在三维空间中，根据物体之间的相对位置和方向，推断出物体之间的关系和相互作用的能力。` |
| **单图**（红图，问「是什么颜色」） | `红色` ✅（`prompt_tokens: 67`，图像 token 被正确编码） |
| **多图**（红+蓝，问「分别是什么颜色」） | `第一张图是红色，第二张图是蓝色。` ✅（`prompt_tokens: 115`） |

多图那次 `prompt_tokens` 从单图的 67 涨到 115，增量与第二张图一致，说明**多图确实都被送进了 vision encoder**，
不是只用了第一张。

## 6. 精度测评

**未完成，但本模型可能不需要。**

赛题 §五 原文明确：部分模型没有官方精度数据「**可以不做**」，列举的正是
`Recon2Reason-Reasoning-4B` 与 `RoboBrain2.5-8B-NV`。

但 T02 的 OCRBench / MMMU 子集评测器与 `data/` 均随分题材料下发，目前未获取，
因此 §6 不做任何数值声明。见 `MISSING.md`。

## 7. 性能参考

6 档 × 2 轮，取 `bench/round2.jsonl`（output tok/s）：

| 档位 | SGLang | vLLM 0.21 | 差值 |
| --- | ---: | ---: | ---: |
| 4K / c1 | 78.35 | 98.83 | vLLM **+26%** |
| 4K / c8 | 441.43 | 544.43 | vLLM **+23%** |
| 4K / c64 | 719.64 | 845.30 | vLLM **+17%** |
| 16K / c1 | 57.86 | 81.93 | vLLM **+42%** |
| 16K / c8 | 226.27 | 267.09 | vLLM **+18%** |
| 16K / c64 | 149.34 | 190.66 | vLLM **+28%** |

**结论：这个模型上 vLLM 全面优于 SGLang（+17% ~ +42%）**，与 MiMo 的结论相反、与 Kimi「高并发 vLLM 反超」部分一致。
可能原因：Recon2Reason 是 4B 小模型，vLLM 的调度开销占比更低；而 MiMo（9B）在 SGLang 的 triton 后端上更吃得开。

**注意**：T02 官方评测器（OCRBench / MMMU）未下发，这些数字是**引擎能力与吞吐证据**，不是官方评分依据。

## 8. 注意事项

- **必须显式 `--attention-backend triton`**（本条目最关键的一条）：不指定时 SGLang 会选 `aiter` backend，
  启动看似成功，但首个 decode 请求就崩：
  ```
  File ".../sglang/srt/layers/attention/aiter_backend.py", line 2535, in forward_decode
      paged_attention_ragged(
  NameError: name 'paged_attention_ragged' is not defined
  ```
  这与 K100-AI 上 `jupyter-sglang` 镜像的 aiter 残包是同一类问题（缺 `aiter.ops.triton.*`）。
- **容器保活与存活检测要分开**：容器用 `tail -f /dev/null` 保活后，`podman ps` 一直是 `Up`，
  光看容器状态会误判服务健康。`container/submit.slurm` 里额外用
  `podman exec ... pgrep -f launch_server` 检测服务进程，否则要白等到超时（实测浪费 30 分钟）。
- **节点磁盘守门是硬需求**：本次先后遇到 `tmp_free=11G`、`0G`，均被 `df` 前置检查 `exit 42` 拦下重投。
- **测试图即时生成而非入库**：纯色 PNG 由脚本生成，代码即规格，不易漂移。
