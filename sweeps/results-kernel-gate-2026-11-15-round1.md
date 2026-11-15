# 闸门结果（第 1 轮，2026-11-15 12:30）—— vLLM 0.20 镜像

> 对应 HANDOFF §3.1 的 0 卡时闸门。作业 `kernel-gate` 1121843，1 卡、`hx1hdnormal`、节点 f11r1n11。
> **本轮探测脚本自身有 bug**：`podman run` 没传 `LD_LIBRARY_PATH`，容器内 `import torch` 报
> `libgalaxyhip.so.5: cannot open shared object file`，所以 Q0/Q1/Q2/Q5 全部失效（不是镜像问题）。
> 有效结论只来自**不需要 import torch 的纯 grep/ls** 部分。修正后重跑见 `results-kernel-gate-2026-11-15.md`。

## 仍然有效的结论

### Q1b：0.20 的 attention backend 模块清单（`ls` 即可，无需 import）

```
cpu_attn.py           flash_attn.py         flex_attention.py    rocm_attn.py
flash_attn_diffkv.py  flashinfer.py         gdn_attn.py          short_conv_attn.py
linear_attn.py        mamba1_attn.py        mamba2_attn.py       triton_attn.py
mamba_attn.py         rocm_aiter_fa.py      rocm_aiter_unified_attn.py
tree_attn.py          turboquant_attn.py    registry.py
```

**判读**：`gdn_attn.py`（Gated Delta Net）、`linear_attn.py`、`short_conv_attn.py` 三者并存，
与 HANDOFF §3.2 A 的判断一致——**Kimi-Linear 走线性注意力路径，与 paged-attention 无关**；
可切换的通用 backend 至少有 `triton_attn` / `flex_attention` / `rocm_attn` / `rocm_aiter_fa`，
以及两个 aiter 变体，所以"换 backend 规避"这条起手动作在 0.20 上是**成立的**。

### Q3：hipBLASLt 的 fp8 线索（部分有效）

- `/opt/dtk/lib/libhipblaslt.so.0.10`（7.3 MB，2026-03-17）存在；
- `strings` 里有 `gfx936`、`gfx936.dat`、`gfx936:sramecc+` —— 该库**包含 gfx936 的调优数据**；
- `hipblaslt-bench --help` 的 fp8 段没能取到（`/opt/dtk/bin/hipblaslt-bench` 不存在或输出被吞），
  且 `import hipblaslt` 因 torch 失败而连带失败 → **fp8 可用性仍未定论**，需要在修好 LD_LIBRARY_PATH 后补测。

### Q4：Kimi-Linear 的 kernel 引用（0.20）

`vllm/model_executor/models/kimi_linear.py` 里只有一处显式 kernel 参数：

```
638: conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"]
```

且镜像内 **没有** `fla` / `fla_core` / `causal_conv1d` / `causal_conv1d_cuda` 任何一个包。
`linear_attn.py` / `gdn_attn.py` / `short_conv_attn.py` 都是 vLLM 自带的 torch-triton 实现，
不依赖 flash-linear-attention。

**这条比 HANDOFF 的假设更进一步**：崩溃现场大概率就在 vLLM 自带的
`short_conv_attn` / `gdn_attn` 里，而不是第三方 `fla_core`。因此"换 fla_core 版本"不是首选，
**换 attention backend 或缩小 buffer 更直接**——与 HANDOFF 给出的起手动作 1/2 吻合。

### Q0/Q1/Q2/Q5：无效

`import vllm` / `import torch` / `import aiter` 全部因 `libgalaxyhip.so.5` 失败。
`/usr/local/lib/python3.10/dist-packages/aiter` **目录存在**（这是 Q2 唯一有效的事实），
说明 0.20 镜像确实装了 aiter 包，只是本轮无法确认其版本与可用算子。

## 闸门判据（按 HANDOFF）

| 判据 | 本轮结果 |
| --- | --- |
| ② 有 aiter | **目录存在**，版本未确认 |
| ① 存在能绕开的 backend | **是**：triton_attn / flex_attention / rocm_attn 可选 |
| 三项全否才停手 | 不适用 —— 未全否 |

→ **闸门未关闭，P0 继续**；但必须先用修好的脚本补一轮 Q0/Q1/Q2/Q5。
