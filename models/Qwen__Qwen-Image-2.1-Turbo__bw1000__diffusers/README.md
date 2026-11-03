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

`PROBE_ONLY=6`（前 6 条真实条件，非自造 prompt）：**6/6 成功，0 失败**，产物 2048×2048 PNG。

| 指标 | 实测 |
| --- | --- |
| 单图耗时 | **~20.7 s**（124 s / 6 张，含 8 步去噪 + VAE 解码 + PNG 落盘） |
| 尺寸 | 2048×2048（与 `conditions/qwen_image.jsonl` 的 `params` 一致） |
| 显存 | 单卡 64 GB 足够，无需 offload |

## 6. 精度测评（官方条件）

官方条件：`conditions/qwen_image.jsonl`，60 输入 × 3 seed（42/2026/1234）= **180 条**，
输出固定 2048×2048，产物 `<id>.png`。**官方不进行统一评分和性能比较**，故不计算 CLIP score / FID。

所有生成参数**直接取自条件清单**（2026-11-03 修正）：

```json
{"width": 2048, "height": 2048, "num_inference_steps": 8,
 "guidance_scale": 1.0, "schedule": "checkpoint", "use_kv_cache": true, "dtype": "bf16"}
```

此前 `evaluate_official.py` 写死 `use_kv_cache=True` 却不传 `num_inference_steps`，
实际按 pipeline 默认 **40 步**跑，与官方条件不符（8 步）。现已逐条读取，见 `docs/gotchas.md` #20。

180 条 × ~20.7 s ≈ **1 h/卡**。

### 关于 2048 vs 1024 的口径

| 来源 | 原文 | 适用 |
| --- | --- | --- |
| 赛事详情 §三 T03 | 「采用 **1024×1024** 作为文生图**性能测试**标准」 | 性能测试 |
| 本次 T03 材料（含条件清单 `params`） | 「输出尺寸固定为 **2048×2048**」 | 质量测试的固定条件 |

**本条目按材料 + 条件清单执行 2048×2048**（机器可读的是 2048，以它为准）。

## 7. 性能参考

官方 T03 明确「不进行统一评分和性能比较」→ `meta.bench = not-required`。
实测单图 **~20.7 s**（2048×2048、8 步、KV cache、单卡 bf16），即约 **0.048 张/s**。

## 8. 注意事项

- **transformers 版本要求不可满足**：官方要 `>=5.17.0`，PyPI 最高 5.9.0。镜像自带 5.3.0 实测可用。
- **必须 `--no-deps` 隔离安装 diffusers**：否则会连带升级 transformers/torch，破坏 DCU 适配。
  但 `--no-deps` **不校验 `Requires-Dist`**，必须自己按元数据对齐版本（见 §9）。
- **`sigmas` 参数**：Turbo 版内置日程，8 步由条件清单给定，勿自行改。
- **显存**：32.5 GB 权重 + 2048×2048 latent，单卡 64 GB 实测够用。

## 9. 阻塞实况 → 已定位并解除（2026-11-03）

| 阶段 | 结果 |
| --- | --- |
| 权重下载 | ✅ 31 GB |
| `QwenImage21Pipeline` | ✅ diffusers 0.41.0 已内置 |
| diffusers 0.42.1 | ❌ 仅支持 Python ≥ 3.11（容器 3.10），且 PyPI 上不可解析 |
| diffusers 0.41.0 + hub 0.36.2 | ❌ `ImportError: cannot import name 'resolve_revision'` |
| **diffusers 0.41.0 + hub 1.33.0** | ✅ 导入通过，`QwenImage21Pipeline OK`，6/6 出图 |

**真实根因**：diffusers 0.41.0 的元数据是 `huggingface-hub>=1.32.0,<2.0`，
而条目用 `pip install --no-deps` 装的是 **hub 0.36.2**（不会报冲突），于是导入即缺符号。
之前「hub 无 `resolve_revision`，属上游未来版本 API」的结论是**错的**：
`resolve_revision` 与 `RevisionResolutionError` 在 hub **1.32.0 / 1.33.0** 里都有（已逐个轮子核对）。
`hub_shim.py` 已删除。

另修一个低级 bug：`container/submit.slurm` 没有把 `MODEL_PATH` 传进容器，
`evaluate_official.py` 里 `os.environ["MODEL_PATH"]` 直接 `KeyError`（此前被导入失败掩盖）。
