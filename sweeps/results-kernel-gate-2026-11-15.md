# 闸门结果（0 卡时，2026-11-15）—— HANDOFF §3.1 的四个问题

> 作业：`kernel-gate` 1121843（第 1 轮，0.20 的 LD_LIBRARY_PATH 有 bug）→ 1121990（修正后重跑）。
> 全部为容器内**只读探测**：不加载模型、不起服务、单卡、`--network=none`。
> 原始输出：`luql:~/work/kernel-gate/{vllm020,vllm021}.md` 与 `console-*-<JID>.log`。

## 结论表

| # | 问题 | vLLM 0.20 | vLLM 0.21 | 判读 |
| --- | --- | --- | --- | --- |
| ① | attention backend 有哪些 | `AttentionBackendEnum` 25 项（含 `TREE_ATTN`）；`RocmPlatform` | `AttentionBackendEnum` 25 项（含 `CUSTOM`，无 `TREE_ATTN`）；`HCUPlatform` | **可切换**，绕开路线成立 |
| ② | aiter 是否存在 | 目录存在；`import aiter` 失败：`Get GPU arch from rocminfo failed ... 'rocminfo'`（容器内无该命令） | 存在，无 `__version__`；导出含 `chunk_gated_delta_rule_fwd`、`asm_fmoe_*`、`awq_gemm_asm` 等 | **有 aiter**，闸门判据 ② 满足（0.20 需补 `rocminfo` 才能 import） |
| ③ | hipBLASLt fp8 | `libhipblaslt.so.0.10` 存在，`strings` 含 `gfx936` / `gfx936.dat` / `gfx936:sramecc+` | 同上（2026-03-04 构建） | **库覆盖 gfx936，但 fp8 算子是否可用仍未验证**（无 bench 二进制、无 python 绑定） |
| ④ | Kimi-Linear 走哪套 kernel | `kimi_linear.py` 仅一处 kernel 参数（`short_conv_kernel_size`）；**无 fla/fla_core/causal_conv1d** | 同上，但 **`causal_conv1d 1.5.4` 已安装** | 走 vLLM 自带 `linear_attn`/`gdn_attn`/`short_conv_attn`，**不是 paged-attn** |

## 三条对 P0 有直接影响的发现

### 1. 0.21 镜像是**海光自研平台插件**，不是 ROCm 平台

```
INFO [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO [__init__.py:46] - hcu -> vllm_hcu:hcu_platform_plugin
INFO [__init__.py:238] Platform plugin hcu is activated
platform class: HCUPlatform
```

`vllm.platforms.rocm` 仍在，但**当前生效的是 `vllm_hcu` 提供的 `HCUPlatform`**。
因此 `get_supported_attn_backends()` / `is_fp8_supported()` 在平台对象上**不存在**
（报 `Current platform cuda does not have ... attribute`），
`supported_quantization` 也不是方法而是 `list`。

**这条纠正了 HANDOFF §3.1 闸门命令的假设**：照着 `vllm/platforms/rocm.py` 去 grep 会一无所获，
要看的是 `vllm_hcu` 插件。`get_device_capability()` 报 `DeviceCapability(major=9, minor=3)`，
即 gfx936 被当作 CUDA 9.3 等价物（这也是 fp8 判定一直不准的原因）。

### 2. Kimi 的线性注意力不依赖 flash-linear-attention

0.20 与 0.21 的 `kimi_linear.py` 都只有一处 kernel 参数：

```
638: conv_kernel_size=hf_config.linear_attn_config["short_conv_kernel_size"]
```

两个镜像都**没有** `fla` / `fla_core`；0.21 多一个**已安装的 `causal_conv1d 1.5.4`**（但 `import` 报
`libc10.so: cannot open shared object file`，需在正确 LD_LIBRARY_PATH 下才能确认可用）。

**对 §3.2 A 的意义**：HANDOFF 里"补 fla_core kernel"的路线**优先级应下调**——
崩溃现场更可能在 vLLM 自带的 `short_conv_attn` / `gdn_attn` 里；
而 0.21 恰好补上了 `causal_conv1d`，**当时判断"直接拿 0.21 跑 Kimi 16K/c64 是最便宜的一枪"**。

> **该判断已于同日被实测推翻（job 1122012）**：0.21 的 `vllm_hcu` 在**加载权重阶段**就崩，
> 四个 rank 全挂，服务起不来。见本文档末尾「实测复核」。**不要再用 0.21 跑 Kimi。**

### 3. `--attention-backend` 对 Kimi **不生效**（已从源码确认，不必再试）

补测（1121990）把 `kimi_linear.py` 的层构造点全抓出来了：

```
21:  from vllm.model_executor.layers.kda import KimiDeltaAttention
177: class KimiMLAAttention(nn.Module)
302:     if config.is_kda_layer(layer_idx):
303:         self.self_attn = KimiDeltaAttention(...)
312:     else:
312:         self.self_attn = KimiMLAAttention(...)
618/634/648: MambaState{Dtype,Shape,CopyFunc}Calculator.kda_state_*()
```

即 Kimi-Linear 的每一层是 **KDA（KimiDeltaAttention）或 MLA（KimiMLAAttention）二选一**，
两者都不是 `AttentionBackend` 的实现（`GDNAttentionBackend` 是另一模型 GDN 的 backend，
`short_conv_attn.py` 也没有注册进 `AttentionBackendEnum`）。

**这条部分成立，但前一版写得过头了**（同日修正）：KDA 层确实不经过选择器，
但 **MLA 层经过**——0.20 的实际启动日志里写着
`[rocm.py:547] Using TRITON_MLA backend out of potential backends: ['TRITON_MLA']`。
所以"换 backend"对 Kimi 的 MLA 那一半是**有效旋钮**，只是候选集当前只有 `TRITON_MLA` 一个。
详见 `sweeps/results-kimi-backend-probe-2026-11-15.md`。

## 实测复核（同日，两发 4 卡）

| 实验 | 作业 | 结果 |
| --- | --- | --- |
| 0.21 跑 Kimi（`--max-model-len 17408`、c64） | 1122012 | ❌ **加载权重阶段即崩**：`vllm_hcu/.../parameter.py:158 → loaded_weight.t()` 遇 3D conv1d 权重报 `t() expects a tensor with <= 2 dimensions, but self is 3D`。4 个 rank 全挂，服务未起 |
| 0.20 的 16K/c64 崩溃现场取证 | 1120618（11-15 10:23 服务） | 崩溃在 Running=29 时；MLA 走 `TRITON_MLA`；`CUDAGraphMode.FULL_AND_PIECEWISE` 因 MLA backend 不支持而降级为 `PIECEWISE` |

**0.21 这条路关闭**。16K/c64 的排查只能留在 0.20 内做。

**0.20 内有效的下一步**（按代价）：
1. `--enforce-eager`（直接证伪"PIECEWISE cudagraph 是元凶"，信息量最高）；
2. `--max-model-len 17408` 排除缓冲越界；
3. MLA backend 候选集溯源 —— **已完成**，见下。

### MLA 候选集只有一个的根因（1122234 + 1122316，0 卡时）

不是框架限制，是**环境缺口的下游后果**：

```
容器无 rocminfo → import aiter 失败 → is_aiter_found_and_supported()=False
  → is_mla_enabled() 被 @if_aiter_supported 短路
  → _get_backend_priorities(use_mla=True) 只返回 [TRITON_MLA]
```

完整证据链见 `sweeps/results-kimi-mla-probe-2026-11-15.md`，坑见 `docs/gotchas.md` #39。
**由这条链推出的第一个动作应当是"补 rocminfo"**，而不是"换 backend"或"改 kernel"。

### 本日两个自伤事故（都记进 gotchas）

| 事故 | 后果 | 记录 |
| --- | --- | --- |
| 编排脚本的 `env.sh` 漏了 `MAX_LEN`，`bench.sh` 在 `set -u` 下直接 abort | 1122249 中 A 配置服务已就绪但压测一行没跑 | 已修：`env.sh` 补齐变量 + 提交前校验 |
| 作业不做 podman 清理，镜像解包留在节点 `/tmp` | 1122426/1122422 双双 `no space left on device`；事后因 `pam_slurm_adopt` **无法删除** | `docs/gotchas.md` #40；所有编排脚本已加 `trap cleanup_podman EXIT` |

## 闸门判定

HANDOFF §3.1 判据：② 有 aiter **或** ① 存在能绕开的 backend → 继续。

- ② **满足**（两个镜像都有 aiter，0.21 还带线性注意力 kernel 入口）
- ① **满足**（0.21 明确列出 25 个 backend，含多个可选实现）

→ **闸门通过，P0 继续投卡时。**

## 对后续顺序的修正建议

| 原计划 | 建议（2026-11-15 实测后修正） |
| --- | --- |
| A：Kimi 16K/c64 先换 backend（0.20） | ~~先换镜像到 0.21~~ → **0.21 关闭**（加载权重即崩，见上）。留在 0.20，按 `--max-model-len 17408` → `--enforce-eager` → MLA backend 候选集 的顺序试 |
| B：fp8 依赖"补 aiter" | 0.20/0.21 都已有 aiter；fp8 的真问题是 `HCUPlatform` 没有 `is_fp8_supported`，要查 `vllm_hcu` 的量化白名单，不是补库 |
| C：MiMo 16K 慢 6.9× | 0.21 的 `HCUPlatform` 可能已改调度，值得复测——但**前提是该模型在 0.21 上能加载**（Kimi 已证明不能，MiMo 需单独确认） |
