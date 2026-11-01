# 环境实测记录

## 硬件

| 卡型 | 实测标识 | 显存 | PwrCap | arch | 位置 |
| --- | --- | --- | --- | --- | --- |
| 异构加速卡 BW（BW1000） | `BW` | 65520 MiB（63.98 GB） | **1000 W** | `gfx936:sramecc+:xnack-` | 华中一区【A区】郑州，Slurm `hx1hdnormal`（58 节点） |
| 异构加速卡 AI（K100-AI） | `K500SM_AI` | 65520 MiB | 300 W | `gfx928:sramecc+:xnack-` | 华东一区【昆山】，仅 Notebook |

节点规格（BW1000）：8 × DCU / 节点，128 CPU，510 GB 内存，IB ×3。
单节点显存总量 **511.9 GB**。

> 赛题口径：`K500-SM 按 K100-AI 计`。`K500SM_AI` 即此卡。

## 软件

| 组件 | SGLang 镜像 | vLLM 镜像 |
| --- | --- | --- |
| 镜像 | `image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1` | `image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0` |
| 压缩大小 | 8.0 GiB（解压 32.2 GB） | 7.8 GiB（解压 29.6 GB） |
| DTK | 26.04（`DTK_VERSION=2604`） | 26.04 |
| torch | `2.9.0+das.opt1.dtk2604`（HIP 6.3.26113） | `2.10.0`（HIP 6.3.26093） |
| sglang | `0.5.10rc0+das.opt2.alpha.dtk2604` | — |
| vLLM | — | `0.20.0` |
| transformers | 5.3.0 | 5.8.1 |
| triton | 3.5.1 | 3.5.0 |
| python | 3.10.12 | 3.10.12 |

镜像仓库 `image.ac.com:5000` 是 **Harbor**，公开项目 13 个、`dcu` 项目下 12555 个仓库，
**匿名可 pull**（`--tls-verify=false`）。SSH 端口全不通，只能 HTTPS。

## 访问路径

| 用途 | 方式 | 说明 |
| --- | --- | --- |
| BW1000 Slurm | `ssh <slurm账号>@zzeshell.scnet.cn:65032` | 需 `hx1hdnormal` 分区权限 |
| K100-AI | SCNet Notebook（OpenAPI `notebook` 接口） | 只有平台账号能建，见 `tools/nb_create.py` |
| 镜像 | `image.ac.com:5000`（内网 10.13.17.207） | 计算节点可达；公网不可达 |
| 权重 | ModelScope 直连 | 登录节点有公网，24 MB/s；计算节点无公网 |

## 镜像内已具备的能力（实测 import 结果）

SGLang 镜像内：

```
OK    sglang.srt.models.kimi_linear        ← Kimi-Linear 无需额外装包
OK    sglang.srt.configs.kimi_linear
OK    sglang.srt.layers.attention.fla.kda  ← KDA kernel 是 vendored 的
OK    sglang.srt.layers.attention.fla.chunk
OK    sglang.srt.layers.attention.fla.fused_norm_gate
FAIL  fla  ModuleNotFoundError             ← 外部 fla 包不需要
OK    einops
```

结论：**官方模型卡要求的 `fla-core ≥ 0.4.0` 在本镜像下不需要**；模板里的 KDA patch 也不需要。

## 资源水位（截至 2026-11-01）

| 项 | 值 |
| --- | --- |
| `/public` 项目配额 | 400 GB |
| 已用 | 141 GB |
| 可用 | 260 GB |
| 其中 Kimi 权重 | 92 GB |
| 其中 MiMo 权重 | 18 GB |
| 其中镜像 tar | 31 GB |
