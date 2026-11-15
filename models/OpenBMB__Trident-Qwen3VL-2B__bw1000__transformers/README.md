# Trident-Qwen3VL-2B on BW1000 × 自建 OpenAI 兼容 serving —— ✅ 跑通（含图文）

## 1. 简介

| 项 | 值 |
| --- | --- |
| 模型 | `OpenBMB/Trident-Qwen3VL-2B`（多模态检索/embedding，Qwen3-VL 主干） |
| 任务 | 文本 / 图像 / 图文 → 同一向量空间（2048 维，L2 归一化） |
| 权重 | 4.0 GB（单 `model.safetensors`，bf16） |
| 卡型 | BW1000 × 1（bf16 权重 ~4 GB，单卡余量充足） |
| 框架 | **自建 serving**：官方 `trident_embedder.py` + 标准库 HTTP 服务（OpenAI 兼容 `/v1/embeddings`） |

**为什么是"自建 serving"而不是 vLLM / SGLang**：检查点 `config.json` 声明
`architectures=["Qwen3VLForEmbedding"]`，且**没有** `auto_map`（权重不自带建模代码），
而 vLLM 0.20 与 SGLang 0.5.10 都**没有注册**这个架构 —— 两条引擎路线均已实测失败并保留证据：

- `../OpenBMB__Trident-Qwen3VL-2B__bw1000__sglang/`（`ValueError: Cannot find model module`）
- `../OpenBMB__Trident-Qwen3VL-2B__bw1000__vllm/`

官方给出的办法是它自己仓库里的 `TridentQwen3VLBackend`（Transformers 直载）。本条目把它落成
一个可复现的 OpenAI 兼容服务：接口对齐 `/v1/embeddings`，客户端不需要改代码即可切换。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（归档于 `images/mimo-sglang-bw10000_1.1.1.tar.gz`） |
| torch | 2.9.0（DCU 版，镜像自带） |
| transformers | **4.57.1**（隔离装到 `/opt/trident-libs`；镜像自带 5.3.0 亦可，见 §6） |
| huggingface_hub | **0.34.0**（隔离装；镜像自带 1.6.0 与 transformers 4.57.1 冲突） |
| qwen_vl_utils | **0.0.14**（版本错会导致图像路径崩，见 §6 第 1 条） |
| 权重路径 | 宿主 `~/models/Trident-Qwen3VL-2B` → 容器 `/models/Trident` |
| 服务端口 | 30000 |

## 3. 目录与脚本

```
env.sh                     环境变量（镜像、权重、wheels、端口）
trident_embedder.py        官方单文件推理封装（MIT，见 THIRD-PARTY.md，未改动）
serve_trident.py           OpenAI 兼容 HTTP 服务（/health、/v1/models、/v1/embeddings）
bench_embed.py             6 档压测（3 输入长度 × 2 并发度），写 bench/summary.md
diag_images.py             图文路径诊断（区分"预处理不一致"与"模型代码问题"）
container/submit.slurm     一键提交（单卡）
container/inner.sh         容器内：装依赖 → 起服务 → smoke → 压测
container/diag.slurm       一次性诊断作业
```

## 4. 一键复现

```bash
sbatch container/submit.slurm                 # 正式：smoke + 压测（ROUNDS=2）
ROUNDS=1 sbatch --export=ALL container/submit.slurm
```

smoke 依次验证三种输入（文本 / 图像 / 图文），并检查向量 **L2 范数 = 1.0**、维度 2048。

## 5. 结果（2026-11-15 实测，单卡 BW1000，bf16）

服务冷启动 **25–35 s**（4 GB 权重）。6 档压测（每档 1 轮；`ROUNDS=2` 可加轮）：

| 输入长度 | 并发 | items/s | tok/s | p50 (ms) | p95 (ms) | 维度 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | **22.787** | 11667 | 43.4 | 43.4 | 2048 |
| 512 | 8 | 23.177 | 11867 | 193.4 | 301.5 | 2048 |
| 2048 | 1 | **15.215** | 31159 | 65.3 | 65.3 | 2048 |
| 2048 | 8 | 15.650 | 32051 | 286.9 | 444.4 | 2048 |
| 4096 | 1 | **8.860** | 36291 | 112.4 | 112.4 | 2048 |
| 4096 | 8 | 9.062 | 37118 | 496.5 | 769.0 | 2048 |

**口径**：客户端并发（每 worker 一条 batch=1 请求），服务端 GPU 侧串行。
`items/s` 是主指标，`tok/s` 按输入 token 估算。

**读法**：并发从 1 抬到 8，吞吐只涨 **1.7%–2.9%**，延迟却涨 4.4×（p50 43→193 ms、112→496 ms）。
说明 2B 模型单次 forward 已经把算力吃满，**吞吐的正确用法是加大单请求 batch**（`EMB_BATCH`），
而不是堆并发；对离线批量索引，建议直接构造多输入请求。

## 6. 两个必须知道的坑

1. **`qwen_vl_utils` 必须 0.0.14**。0.0.8 的预处理与 Qwen3-VL 的 grid/patch 约定不一致，
   任何图像输入都会在模型里崩：
   `RuntimeError: shape '[1, 1, 2, 3, 32, 2, 16, 32, 2, 16]' is invalid for input of size 6439776`。
   诊断方法见 `diag_images.py`（它同时打印 processor 的 `pixel_values` 与 `image_grid_thw`，
   用于区分"预处理不一致"和"模型代码算错"）。纯文本路径不受影响，**所以只测文本会误判为"已跑通"**。
2. **镜像自带 `huggingface_hub` 1.6.0**，而 transformers 4.57.1 要求 `<1.0`，导入即
   `ImportError: huggingface-hub>=0.34.0,<1.0 is required`。解法是隔离装 0.34.0 到 `/opt/trident-libs`
   并把 `PYTHONPATH` 顶到最前（本条目已这么做）。

另外：图像边长要能被 16 整除、且不小于 `min_pixels`（256×32×32）。320×240 这类小图会触发
同类 reshape 报错，测试图请用 1024×1024（`assets/red1024.png`）。

## 7. 性能小结与适用面

| 场景 | 结论 |
| --- | --- |
| 离线批量建索引（长文本） | 4K 档 8.9 条/s、36.3K tok/s；单条 112 ms |
| 在线召回（短 query） | 512 档 22.8 条/s、p50 43 ms；并发几乎不增益 → 单实例够用 |
| 图文混合检索 | 支持（同一向量空间），但每条要过视觉塔，吞吐低于纯文本 |

## 8. 与其它条目的关系

- 本条目的价值是**补上"两引擎都不支持"这一类模型的第三条路**：模型自带单文件推理代码时，
  用"官方实现 + 最小 HTTP 封装"就能得到一个与 vLLM/SGLang **平行**的运行时可选项，
  而且不需要改任何 kernel。
- 两个 blocked 条目（sglang / vllm）**保留不动**：它们记录的是"框架不支持"这一事实与证据，
  与本条目的"绕开框架"是两条独立结论。
- T02 官方评测器与 `baselines/` 至今未下发，故本条目只给**吞吐与链路验证**，不给精度分数
  （`meta.json` 的 `bench_note` 已写明）。
