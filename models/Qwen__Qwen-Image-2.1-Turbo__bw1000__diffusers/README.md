# Qwen-Image-2.1-Turbo on BW1000 + Diffusers —— 🟡 进行中

> **状态：官方评测作业已提交，等待集群资源。** 本文件先记录已确认的事实与已知风险。

## 1. 简介

`Qwen/Qwen-Image-2.1-Turbo` 在 `异构加速卡BW（BW1000）` + `Diffusers`（bf16）上的适配。

通义千问 Qwen-Image-2.1 的加速版 checkpoint，支持文生图与图像编辑，只需 8 步去噪。
checkpoint 内置推荐 8 步采样日程，默认 CFG=1，并通过前缀 KV 缓存复用文本与参考图上下文。
Diffusers 目录结构（`transformer` / `text_encoder` / `vae` / `scheduler` / `processor`），
共 16.2B 参数（transformer 7.1B、text_encoder 8.8B、VAE 0.34B），全 BF16 约 32.5 GB，Qwen Research License。

**与官方部署方式的差异（已确认的坑）**：

1. 官方模型卡要求 `transformers>=5.17.0`，但 **PyPI 上 transformers 最高只有 5.9.0**，
   该版本不存在。本条目用镜像自带 5.8.1 实测。
2. 官方建议从源码装 Diffusers（需 PR #14950）。实测 **diffusers 0.42.1 已内置 `QwenImage21Pipeline`**，
   不必从源码装。`--no-deps` 装进隔离目录即可。
3. **只设 `num_inference_steps` 不会覆盖内置 sigmas 日程**，需显式传 `sigmas`（本条目暂用默认日程，
   并作为 `parameter_changes` 记录）。

## 2. 验证环境

```text
image: image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
torch: 2.9.0+das.opt1.dtk2604   triton: 3.5.1   transformers: 5.3.0（镜像自带）
diffusers: 0.42.1（隔离安装于 /opt/qi-libs）
Device: BW × 1，64.0 GB，gfx936
```

## 3. 适配优化方案

### 3.1 基础镜像

用 SGLang 镜像（自带 DCU 版 torch 与完整视觉栈）。

### 3.2 软件包升级

```bash
pip install -q --no-deps --target /opt/qi-libs diffusers-0.42.1-py3-none-any.whl
export PYTHONPATH=/opt/qi-libs:$PYTHONPATH
```

隔离安装，不动镜像内 torch / transformers（避免破坏 DCU 适配版）。

### 3.3 代码改动

无 patch。

### 3.4 运行步骤

```bash
PROBE_ONLY=3 sbatch --export=ALL container/submit.slurm   # 先跑 3 条验证管线
sbatch container/submit.slurm                             # 全量 180 条
```

## 4. 运行方式

一次性批处理，无常驻服务。入口 `container/submit.slurm`。

## 5. Smoke 测试

⚠️ **待填**（等探针作业完成）。`PROBE_ONLY=3` 会先跑 3 条并输出尺寸与耗时。

## 6. 精度测评（官方条件）

⚠️ **待填**。官方条件：`formal/conditions/qwen_image.jsonl`，60 输入 × 3 seed（42/2026/1234）= **180 条**，
输出**固定 2048×2048**，产物按 `<id>.png` 命名。

**官方不进行统一评分和性能比较**，因此本条目不计算 CLIP score / FID。

### ⚠️ 一个需要官方确认的口径冲突

| 来源 | 原文 | 适用 |
| --- | --- | --- |
| 赛事详情 §三 T03 | 「采用 **1024×1024** 的图片输出作为文生图的**性能测试**标准」 | 性能测试 |
| 本次 T03 材料 | 「**输出尺寸固定为 2048×2048**，输出图像」 | 质量测试的固定条件 |

两者一个说性能、一个说质量，可能并不矛盾。**本条目按材料要求用 2048×2048**，
若官方另有口径需调整尺寸并重跑。

## 7. 性能参考

官方 T03 明确「不进行统一评分和性能比较」→ `meta.bench = not-required`。
单张 2048×2048 的生成耗时待测（Turbo 版为 8 步 + KV cache）。

## 8. 注意事项

- **transformers 版本要求不可满足**：官方要 `>=5.17.0`，PyPI 最高 5.9.0。用 5.3.0（镜像自带）实测，
  若出现 API 不兼容需记录。
- **必须 `--no-deps` 隔离安装 diffusers**：否则会连带升级 transformers/torch，破坏 DCU 适配。
- **2048 vs 1024 口径待确认**（见 §6）。
- **`sigmas` 参数**：Turbo 版内置日程，只改 `num_inference_steps` 无效。
- **显存**：32.5 GB 权重 + 2048×2048 的 latent，单卡 64 GB 需实测是否够；
  不够则需 `enable_model_cpu_offload()` 或降分辨率（后者会偏离官方条件，须记入 `parameter_changes`）。
