# 量化分支：已做的实测、剩余成本与建议（2026-11-15）

> 结论先行：**量化不是"没试过"，而是"试过并被数据否掉了一半"**。本文把已有的硬事实集中在一处，
> 便于下一次决定要不要继续投入，而不是从头再摸索一遍。

> **2026-11-15 补充（作业 1123947，决定性）**：fp8 的问题**不是配置，是 kernel 不存在**。
> `torch.mm` 对 `float8_e4m3fn` / `float8_e5m2` 直接抛
> `NotImplementedError: "addmm_cuda" not implemented`。
> 白名单有 `fp8`、`Fp8Config` 能解析、`fp8_dtype` 有定义、hipBLASLt 含 `R_8F_E4M3` 符号——
> **全都对，但 torch 的分发层没有 fp8 GEMM 实现**。
> 详细证据与"为什么表现为初始化失败"见 `sweeps/results-kimi-quantigate-2026-11-15.md`
> 与 `docs/gotchas.md` #46。
> **这意味着在 torch 侧补齐 fp8 GEMM 绑定之前，任何 fp8 自量化都不可能成功**，
> 不存在"调参数/换版本"的解法。量化分支应视为**外部阻塞**，不是待办。

## 1. 已经验过的三条路（都有产物或日志）

| 路径 | 结果 | 证据 |
| --- | --- | --- |
| 现成 AWQ 权重（MiMo-9B） | ✅ 能起服务、能跑完，但 **慢 5×** | `models/XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm/`、`sweeps/results-awq-2026-11-06.md` |
| `--quantization fp8`（Qwen3-ASR） | ❌ engine core 初始化即失败：`torch.addmm` 无 fp8 实现 | `docs/gotchas.md` #22、#46；`sweeps/results-kimi-quantigate-2026-11-15.md` |
| 官方 AWQ / GPTQ 权重 | ❌ MiMo、Kimi 都**没有**官方量化权重 | 各条目 `MISSING.md` |

**为什么 AWQ 会慢 5×**：vLLM 在 DCU 上走的是"int4 权重 → 反量化到 bf16 → bf16 GEMM"的回退路径，
省了显存但**没省算力**，反而多了一次反量化访存。这与"缺 fused dequant-GEMM kernel"是同一个根因
（见 `docs/gotchas.md` #21 / #22）。所以在 kernel 补齐之前，**自量化不会带来吞吐收益**，
只能换来"显存更省、能装下更大模型"这一个好处。

## 2. 如果要走"自量化"，需要先补齐的材料（本次已探明）

| 项 | 现状 |
| --- | --- |
| 工具链 | `llmcompressor 0.15.0` 在 PyPI 可得；它要求 `compressed-tensors==0.20.1`、`pillow>=12`（镜像自带 11.3，需隔离装） |
| 校准数据 | **HuggingFace 在集群不可达**（`curl https://huggingface.co` 无响应），`ultrachat_200k` 等内置数据集拿不到 → 必须用本地语料自建校准集（可直接用 `official/t03-t04/conditions/*.jsonl` 里的 prompt，约 700 条，足够） |
| wheel 兼容 | 容器是 **py3.10**，登录节点是 **py3.12** → 离线 wheelhouse 必须用 `pip download --python-version 310 --platform manylinux2014_x86_64 --abi cp310 --only-binary :all:` 打（这条已在 `docs/gotchas.md` #18 记录过） |
| 依赖链长度 | llmcompressor 的依赖树里 `compressed-tensors → transformers/accelerate/datasets`，与镜像自带的 transformers 5.3 有版本张力，需要一次性把整棵树的 cp310 轮子凑齐，否则每轮试错要 25 min（一次作业） |

## 3. 成本-收益（按现在的实测数据推算）

| 方案 | 卡时 | 预期收益 |
| --- | --- | --- |
| MiMo-9B → AWQ W4A16（自量化） | 2–3（量化 1 + 压测 1–2） | 显存 9 GB → 约 5 GB；**吞吐预计仍是 bf16 的 1/5**（反量化回退路径不变） |
| Kimi-48B → AWQ | 4–8 | 同上，且 48B 量化后仍要 TP=2~4 |
| 补 fused dequant-GEMM kernel（自建镜像） | 8–16 | 这才是量化的**收益前提**；没有 kernel，一切量化都只是省显存 |

## 4. 建议

1. **短期不建议做自量化**：在当前 kernel 条件下它只能"省显存"，不能"提吞吐"，而本项目的 19 个条目
   在 bf16 下都已经能装进单卡或 4 卡，省显存换不来新能力。
2. **要做的顺序反过来**：先解决 `docs/gotchas.md` #21（paged-attn）/ #22（fp8 GEMM）这类 kernel 缺口，
   量化才有意义；那时自量化的一次性投入（2–3 卡时）才划算。
3. 如果下一步一定要"补齐全库最大空白"，建议先做 **1 卡时的可行性 spike**：
   只做"llmcompressor 能否在镜像里 import + 用本地校准集量化一个 0.5B 模型"这一步，
   验证依赖树齐了再投 9B/48B 的量化时间。

## 5. 本次实际动作记录

- 检查了 PyPI 可得性：`llmcompressor==0.15.0` ✅
- 检查了校准数据来源：HF 不可达 ❌ → 需本地语料
- 未提交量化作业（避免在 kernel 缺口未解决时消耗 2–3 卡时换一个必然的负结果）
