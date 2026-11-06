# HY-OmniWeaving on BW1000 × 官方 hyvideo 代码 —— 🟡 进行中

## 1. 简介

| 项 | 值 |
| --- | --- |
| 模型 | `Tencent-Hunyuan/HY-OmniWeaving`（MLLM + MMDiT + VAE 的统一视频生成框架） |
| 任务 | T03 文生视频：`prompt` → 832×480 / 81 帧 / 16 fps |
| 权重 | 主检查点 55.3 GB（transformer 32 G + text_encoder ckpt 16 G + vae 4.7 G + upsampler 274 M） |
| 另有依赖权重 | Qwen2.5-VL-7B（16.6 G）、byt5-small（3.6 G）、Glyph-SDXL-v2（1.9 G）、FLUX.1-Redux siglip（1.1 G） |
| 卡型 | BW1000 × N（`WORLD_SIZE` 序列并行；N=1 也可跑） |

**与官方部署方式的差异**：

1. 官方要求 `flash_attn`；镜像自带的那份**可用但 API 不同**——`unpad_input` 返回 4 项，
   而上游按 5 项解包（`ValueError: not enough values to unpack (expected 5, got 4)`）。
   为此加了 `patches/0001-flash-attn-unpad-compat.patch`（运行时兼容两种 API），**不改模型逻辑**。
2. 官方 requirements 里 `angelslim==0.2.1` 在 PyPI 已升到 0.5.0 且要求 transformers>=5.2，
   与本条目的 transformers 4.57.1 冲突 → 推理路径不需要它，**不安装**。
3. 官方 byT5 走 `google/byt5-small` + `Glyph-SDXL-v2`；两者都在 ModelScope 有镜像，
   条目按官方目录结构落到 `text_encoder/{llm,byt5-small,Glyph-SDXL-v2}`、`vision_encoder/siglip`。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（digest 见 `image.lock`） |
| torch / torchvision | 2.9.0+das / 镜像自带 |
| transformers / diffusers / hub | 4.57.1 / 0.32.2 / 0.34.0（隔离装到 `/opt/omni-libs`） |
| 卡 | BW × 1（`gfx936:sramecc+:xnack-`，64 GiB） |
| 权重路径 | 宿主 `~/models/HY-OmniWeaving` → 容器 `/models/Omni` |

## 3. 适配优化方案

### 3.1 基础镜像

与 Wan2.1 同一个 SGLang 镜像（唯一自带 DCU 版 torch 2.9 + torchvision 的镜像），**不自建**。

### 3.2 软件包升级

16 个 cp310 轮子隔离装到 `/opt/omni-libs`（清单见 `requirements-extra.txt`），
不覆盖镜像自带的 torch/transformers。刻意不装 `angelslim`、`flash-attn`、`SageAttention`、
`Flex-Block-Attention`（后三者是可选加速）。

### 3.3 代码改动或参数

| 改动 | 原因 |
| --- | --- |
| `patches/0001-flash-attn-unpad-compat.patch` | 镜像 flash_attn 的 `unpad_input` 返回 4 项，上游按 5 项解包 → `ValueError`。补丁加 `_unpad_compat()` 兼容两种签名（**不改注意力数学**） |
| `--offloading true`（默认） | 32 G DiT + 16 G MLLM 单卡装不下，走 CPU offload 逐组搬运 |
| `think=False` | 官方条件 `params.think=false`，不启用 MLLM thinking 模式 |
| **`--aspect_ratio 832:480`（不能用 16:9）** | pipeline 按比例从 121 个尺寸桶里挑，`16:9` 会落到 **848×480**；`832:480` 才能精确命中官方要求的 832×480。**首轮 60 条因此全是 848×480，已作废重跑** |

### 3.4 升级步骤

```bash
sbatch container/submit.slurm                 # 官方 60 条
PROBE_ONLY=2 sbatch container/submit.slurm    # 只跑 2 条
```

## 4. 服务启动脚本

无常驻服务（一次性批处理）。`container/inner.sh`：装依赖 → 复制源码并打补丁 →
导入校验（含 `hyvideo`）→ `evaluate_official.py` 逐条调用官方 `generate.py` → 产物规格校验。

## 5. Smoke 测试

两轮探针（2026-11-04 夜，单卡）：

| 探针 | 结果 |
| --- | --- |
| 17 帧 / 8 步 | ✅ 出片 101 KB，含 16 个轮子安装与模型加载共 **3.6 min** |
| **81 帧 / 50 步（= 官方条件）** | ✅ 出片 **285 KB**，从容器启动到结束 **26 min**（含模型加载与 CPU offload 搬运） |

显存：单卡 64 GB 够（32 G DiT + 16 G MLLM 走 offloading 分组搬运，未 OOM）。

## 6. 精度测评

官方条件：`conditions/omniweaving.jsonl`，**60 条**（20 prompt × seed 42/2026/1234）。
参数逐条取自清单：832×480 / 81 帧 / 16 fps / 50 步 / guidance 6.0 / flow_shift 7.0 / bf16。
官方明示不评分 → 只交原始产物 + 规格统计。

⚠️ 待跑。

## 7. 性能参考

官方 T03 不评分 → `meta.bench = not-required`。

| 场景 | 实测 |
| --- | --- |
| 单条（81 帧 / 480p / 50 步，单卡） | **26 min**（含模型加载；offloading 开启） |
| 60 条预估（单卡） | ~26 h → 因此改用 4 卡序列并行（`WORLD_SIZE=4`）跑全量 |

## 8. 注意事项

- **依赖链最长的一个条目**：主检查点之外还要 4 份权重（Qwen2.5-VL-7B、byt5-small、
  Glyph-SDXL-v2、FLUX.1-Redux siglip），缺任何一份都会在模型初始化阶段报错
  （第一次踩的是 `RuntimeError: Error loading byT5 glyph processor`）。
- **FLUX.1-Redux-dev 在 HuggingFace 是 gated 模型**（需申请访问）；我们用 ModelScope 镜像
  `AI-ModelScope/FLUX.1-Redux-dev`，内容是 siglip 视觉编码器（1.1 G）。
- 官方源码里的 `SyntaxError: f-string expression part cannot include a backslash` 是
  `generate.py` 打印帮助文本时触发的**无关警告**，不影响执行。
