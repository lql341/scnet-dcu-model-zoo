# Kimi-Linear-48B-A3B-Instruct on BW1000 × **vLLM 0.20** —— ✅ 6 档（5 档有效）

> 与同目录 `moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang` 配对：
> **同一模型、同一张卡、只换引擎**，用于跨框架硬对比。

## 1. 简介

| 项 | 值 |
| --- | --- |
| 模型 | `moonshotai/Kimi-Linear-48B-A3B-Instruct`（48B 总参 / 3B 激活，线性注意力混合） |
| 权重 | 92 GB BF16（20 分片） |
| 卡型 | BW1000 × 4（TP=4 → 约 23 GB/卡） |
| 引擎 | vLLM 0.20.0 |

**与官方部署方式的差异**：无。官方模型卡本身给出 vLLM / SGLang 两种用法。

### 为什么这个条目值得建（一条被推翻的旧结论）

`HANDOFF.md` 早期记录过「vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio），
所以 Kimi 在 vLLM 上 blocked」。**2026-11-12 实测推翻**：

```python
from vllm.model_executor.models.registry import ModelRegistry
archs = ModelRegistry.get_supported_archs()      # 345 个架构
[k for k in archs if "kimi" in k.lower()]
# → ['KimiK25ForConditionalGeneration', 'KimiLinearForCausalLM',
#    'KimiVLForConditionalGeneration', 'MoonshotKimiaForCausalLM']
```

`KimiLinearForCausalLM` 在列 —— 架构是支持的，之前记的 blocked 是误判（可能当时看的是旧版本或只 grep 了文件名）。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`（digest `sha256:07930260…`） |
| torch / vLLM | 2.10.0+das / 0.20.0 |
| transformers / triton | 5.8.1 / 3.5.0 |
| 卡 | BW × 4（TP=4，约 23 GB/卡） |
| 端口 | 127.0.0.1:30000 |
| 权重路径 | 宿主 `~/models/Kimi-Linear-48B-A3B-Instruct` → 容器 `/models/Kimi` |
| 冷启动 | 约 640–700 s（含 vLLM 对线性注意力的 kernel 编译） |

## 3. 适配优化方案

### 3.1 基础镜像

`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`
（torch 2.10 / vllm 0.20 / transformers 5.8.1；与 MiMo×vLLM 条目共用）

### 3.2 软件包升级

无。Kimi-Linear 由 vLLM 内置实现承载，`--trust-remote-code` 用于模型自带的配置/分词器代码。

### 3.3 代码改动或参数

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--tensor-parallel-size` | 4 | 92 GB 权重，单卡放不下 |
| `--max-model-len` | 20480 | 16K 输入 + 1K 输出 + chat template |
| `--gpu-memory-utilization` | 0.92 | vLLM 在 DCU 上探测偏保守，0.85 会报 cache blocks 不足 |
| `--max-num-seqs` | 128 | 与最高并发档对齐 |
| **不启用** `--enforce-eager` | — | CUDA Graph 在 MiMo/MiMo-SGLang 上实测快 3%~32%，本条目同样保持开启 |

### 3.4 升级步骤

```bash
sbatch container/submit.slurm        # 起服务 + smoke + verify + 6 档×2 轮压测
```

## 4. 服务启动脚本

`container/submit.slurm` 双镜像编排：vLLM 镜像起服务（`--network=host`），SGLang 镜像跑官方压测客户端
（vLLM 镜像不含 `sglang.benchmark.serving`）。

## 5. Smoke 测试

| 检查项 | 结果 |
| --- | --- |
| 架构识别 | ✅ vLLM 用内置 `KimiLinearForCausalLM` 加载（无需自定义模型代码） |
| 服务启动 | ✅ `/health` 通；engine core 正常（多卡必须显式 `-e HIP_VISIBLE_DEVICES=0,1,2,3`） |
| 生成 | ✅ 4K/c1 TTFT 407 ms、TPOT 24.2 ms，输出通顺 |
| `verify.py` | `selfcheck OK`；精度部分 `[BLOCKED]`（缺官方 `baselines/`） |

## 6. 精度测评

T01 官方 `baselines/<模型>.json` 从未下发 → **[BLOCKED]**，不填估算值（见 `MISSING.md`）。

## 7. 性能参考

取 `bench/round2.jsonl`（单位 output tok/s），与同模型 SGLang 条目并列：

| 档位 | **vLLM 0.20** | SGLang 0.5.10 | vLLM 相对 |
| --- | ---: | ---: | ---: |
| 4K / c1 | 40.73 | 104.67 | −61% |
| 4K / c8 | 304.18 | 353.53 | −14% |
| 4K / c64 | **858.08** | 839.00 | **+2%** |
| 16K / c1 | 37.36 | 50.80 | −26% |
| 16K / c8 | 197.83 | 119.21 | **+66%** |
| 16K / c64 | ❌ 无数据 | 234.17 | — |

**结论（与 MiMo 的"vLLM 全面落后"不同，值得记录）**：
- **高并发档 vLLM 反超**：4K/c64 +2%、16K/c8 **+66%**。Kimi 是 48B 小激活（3B）MoE-like 结构，
  vLLM 的调度在高并发下更吃得开。
- **单流 vLLM 明显落后**（4K/c1 −61%），说明低并发时 SGLang 的 kernel 更优。
- **16K/c64 两轮均失败**：压测客户端一条请求都没成功（`IndexError: ... size 0`），
  同 MiMo×vLLM 的症状，根因是 DCU 缺 paged-attention 内核（`docs/gotchas.md` #21）。

### 7.1 多卡起服务必须显式给可见性

Slurm 只把 1 张卡写进 `CUDA_VISIBLE_DEVICES`，容器默认继承后 vLLM 只看到 1 张卡，
TP=4 直接 `Engine core initialization failed`。必须：

```bash
podman run ... -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3
```

这条已写入 `container/submit.slurm`，也是 gotchas #21 的补充。

## 8. 注意事项

- 首次起服务会触发 vLLM 对 Kimi 线性注意力的 kernel 编译，冷启动可能 >10 min（`LOAD_TIMEOUT` 设 2400 s）
- 本机 NFS 上 `cp -rf` 会漏拷/不覆盖文件 → 同步一律用 `rsync`（gotchas #26）
