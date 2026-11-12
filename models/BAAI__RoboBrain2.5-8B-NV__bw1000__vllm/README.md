# RoboBrain2.5-8B-NV on BW1000 × **vLLM 0.21** —— ✅ 6 档完成（跨引擎证据）

> 与同目录 `BAAI__RoboBrain2.5-8B-NV__bw1000__sglang` 配对：同一模型、同一卡，**只换引擎**。

## 1. 简介

| 项 | 值 |
| --- | --- |
| 任务 | T02 视觉多模态理解（图片问答/推理） |
| 架构 | `Qwen2_5_VLForConditionalGeneration` —— vLLM 0.20 的 ModelRegistry 里**有**（2026-11-12 实测，345 个架构） |
| 卡型 | BW1000 × 1（TP=1，权重 ~9 GB BF16） |

**与官方部署方式的差异**：无。Qwen2.5-VL 系列是 vLLM 一等公民。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `dcu/wqxyyt06/base/vllm0.21.0-ubuntu22.04-dtk2604-py3.10:v1`（digest `sha256:c6a2f554…`） |
| 框架 | **vLLM 0.21.0** |
| 卡 | BW1000 × 1（TP=1） |
| 端口 | 127.0.0.1:30000 |
| 冷启动 | 200–260 s |

## 3. 适配优化方案

### 3.1 基础镜像

`dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`（torch 2.10 / vllm 0.20 / transformers 5.8.1）

### 3.2 软件包升级

无。

### 3.3 代码改动或参数

| 参数 | 值 | 理由 |
| --- | --- | --- |
| `--max-model-len` | 32768 | 与 SGLang 条目一致，留足多图 token |
| `--gpu-memory-utilization` | 0.95 | DCU 上 vLLM 探测偏保守 |
| `--max-num-seqs` | 128 | — |
| **不启用** `--enforce-eager` | — | 与其它 vLLM 条目保持一致（CUDA Graph 实测更快） |

### 3.4 升级步骤

```bash
sbatch container/submit.slurm
```

## 4. 服务启动脚本

双镜像编排：vLLM 镜像起服务（`--network=host`），SGLang 镜像跑官方压测客户端。

## 5. Smoke 测试

| 检查项 | 结果 |
| --- | --- |
| 服务启动 | ✅ `/health` 200 |
| 纯文本 | ✅ 正常返回 |
| **单图** | ✅ 正常返回（vLLM 0.20 上此步直接 500，见 §8） |
| **多图（2 张）** | ✅ 正常返回 |
| `verify.py` | `selfcheck OK`；精度 `[BLOCKED]`（T02 评测器未下发） |

## 6. 精度测评

T02 官方评测器（OCRBench / MMMU）未下发 → **[BLOCKED]**。本条目目标是**「vLLM 侧跑通」**，
与 `__sglang` 条目配对构成跨引擎证据，不产出精度分数。

## 7. 性能参考

6 档 × 2 轮，取 `bench/round2.jsonl`（单位 output tok/s）：

| 档位 | token/s | TPOT (ms) |
| --- | ---: | ---: |
| 4K / c1 | **64.01** | 10.1 |
| 4K / c8 | **387.93** | 14.5 |
| 4K / c64 | **667.69** | 70.1 |
| 16K / c1 | **51.14** | — |
| 16K / c8 | **225.76** | — |
| 16K / c64 | **175.71** | — |

**注意**：T02 官方评测器（OCRBench / MMMU）与 `baselines/` 均未下发，所以这些数字是
**引擎能力与吞吐证据**，不是官方评分依据。完整 6 档明细见 [`bench/summary.md`](bench/summary.md)。

## 8. 注意事项

### 8.1 为什么必须用 vLLM 0.21（一条硬约束）

vLLM **0.20** 的 `qwen3_vl.py::_get_deepstack_input_embeds` 会硬性断言，任何带图请求都 500：

```
ValueError: Requested more deepstack tokens than available in buffer: num_tokens=72 > 67
```

根因：deepstack 缓冲按**首个视觉请求的 token 数**分配，后续请求 token 更多就越界。
**0.21 已删除该断言**（改为直接切片），实测单图/多图均正常。
这也说明「SGLang 能跑、vLLM 不能」往往是**版本问题**而不是能力问题 —— 换镜像比改模型划算。

### 8.2 其它

- 多卡时（TP>1）必须显式 `-e HIP_VISIBLE_DEVICES=0,1,...`，否则容器只见 1 张卡（gotchas #21）
- 本机 NFS 上 `cp -rf` 会漏拷/不覆盖 → 同步一律用 `rsync`（gotchas #26）
- 收容器前要把 `podman logs` 落盘，否则 500 的根因随容器消失（本条目踩过：一开始只留了 1 行 digest）
