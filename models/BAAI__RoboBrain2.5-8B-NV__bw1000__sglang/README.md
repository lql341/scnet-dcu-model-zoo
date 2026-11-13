# RoboBrain2.5-8B-NV on BW1000 × **SGLang 0.5.10** —— ✅ 6 档完成（与 vLLM 配对）

> T02 视觉多模态理解。本条目验证「图像预处理 → vision encoder → LLM」全链路，含**多图**输入。

## 1. 简介

`BAAI/RoboBrain2.5-8B-NV` 在 `异构加速卡BW（BW1000）` + `SGLang 0.5.10`（bf16）上的适配验证。

智源（BAAI）的**具身大脑**模型 RoboBrain 2.5 的 8B 版（2026-01 发布，技术报告 arXiv 2601.14352）。
在 2.0 的感知、推理与规划能力上用大规模时空数据继续训练，空间推理从 2D 升级到 3D：
预测带深度的 3D 坐标点、理解绝对度量约束、输出关键点序列形式的 3D 空间轨迹；
时序上引入通用奖励模型（GRM）做稠密进度预测。

`config.json` 实测：`architectures=['Qwen3VLForConditionalGeneration']`、`model_type=qwen3_vl`，
结构与 Qwen3-VL-8B 相同（语言模型 36 层、hidden 4096、32 头 / 8 KV 头，视觉编码器 27 层），
BF16 权重 2 分片约 17 GB，Apache-2.0。

同系列另有 `-MT`（摩尔线程集群训练）与 4B 版，架构一致。本件用 `-NV` 版。

**与官方部署方式的差异**：官方 README 用 Transformers 直接加载（`transformers==4.57.1`、`torch>=2.6`）。
本件改用 SGLang 提供 OpenAI 兼容服务，唯一必要的适配是**显式指定 `--attention-backend triton`** ——
本镜像默认会选中 `aiter` attention backend，而该 backend 残缺，启动后立即崩（见 §8、`docs/gotchas.md`）。

模型是标准 `Qwen3VLForConditionalGeneration`，**无需 `trust_remote_code` 的自定义代码、无需装包、无需 patch**。

- 权重：`https://www.modelscope.cn/models/BAAI/RoboBrain2.5-8B-NV`
- 镜像：见 `image.lock`
- 提交仓库：https://developer.sourcefind.cn/codes/luql/robobrain25-8b-nv-bw1000-sglang

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
- 冷启动：**200 s**
- 模型路径（容器内）：`/models/RoboBrain`
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
  --model-path /models/RoboBrain --served-model-name robobrain2.5-8b-nv \
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
| `/v1/models` | `robobrain2.5-8b-nv`，`max_model_len: 32768` |
| 纯文本 | `空间推理是指通过分析和理解物体在空间中的位置、形状、大小、方向等关系，来推断和解决问题的思维过程。` |
| **单图**（红图，问「是什么颜色」） | `红色` ✅（`prompt_tokens: 67`，图像 token 被正确编码） |
| **多图**（红+蓝，问「分别是什么颜色」） | `这两张图分别展示了两种纯色：* **第一张图**：是**红色**。* **第二张图**：是**蓝色**。` ✅ |

多图那次 `prompt_tokens` 从单图的 67 涨到 115，增量与第二张图一致，说明**多图确实都被送进了 vision encoder**，
不是只用了第一张。

## 6. 精度测评

**未完成，但本模型可能不需要。**

赛题 §五 原文明确：部分模型没有官方精度数据「**可以不做**」，列举的正是
`RoboBrain2.5-8B-NV` 与 `RoboBrain2.5-8B-NV`。

但 T02 的 OCRBench / MMMU 子集评测器与 `data/` 均随分题材料下发，目前未获取，
因此 §6 不做任何数值声明。见 `MISSING.md`。

## 7. 性能参考

6 档 × 2 轮，取 `bench/round2.jsonl`（output tok/s）：

| 档位 | SGLang | vLLM 0.21 | 差值 |
| --- | ---: | ---: | ---: |
| 4K / c1 | 45.69 | 64.01 | vLLM **+40%** |
| 4K / c8 | 327.14 | 387.93 | vLLM **+19%** |
| 4K / c64 | 543.80 | 667.69 | vLLM **+23%** |
| 16K / c1 | 37.14 | 51.14 | vLLM **+38%** |
| 16K / c8 | 126.65 | 225.76 | vLLM **+78%** |
| 16K / c64 | 141.66 | 175.71 | vLLM **+24%** |

**结论：与同族的 Recon2Reason 一致 —— vLLM 全面优于 SGLang（+19% ~ +78%）。**
两个 Qwen3-VL 小模型都呈现这个规律，说明不是偶然：
小模型下 vLLM 的调度优势压过了 SGLang 的 kernel 优势。

（对照：MiMo-9B 上 SGLang 领先 1.5–7 倍；Kimi-48B 上则是「单流 SGLang 赢、高并发 vLLM 赢」。）

**注意**：T02 官方评测器未下发，这些数字是引擎能力证据，不是官方评分。

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
