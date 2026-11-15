# P0-B 量化闸门：fp8 在 BW1000 上**没有 GEMM kernel**（2026-11-15，作业 1123947）

> 0 卡时探测（1 DCU，纯 import + 最小 GEMM，不加载模型）。
> 目标：回答 HANDOFF §3.2 B 的问题——fp8 到底卡在哪，是缺库、缺白名单，还是缺 kernel。

## 结论先行

**`torch.mm` 对 fp8 直接 `NotImplementedError`。** 不是白名单问题，不是库缺失，是**没有 kernel**。

```python
torch 2.10.0   device BW
torch.float8_e4m3fn  mm  FAIL  NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'
torch.float8_e5m2    mm  FAIL  NotImplementedError: "addmm_cuda" not implemented for 'Float8_e5m2'
```

`torch.addmm` 是 PyTorch 分发到后端 GEMM 的唯一入口。它报 `NotImplementedError`
意味着 **torch 的 DCU/HIP 后端没有为 fp8 dtype 注册任何矩阵乘实现**，
而不是"调用了较慢的路径"。任何上层（vLLM、hipBLASLt、aiter）都无法在它之上凭空造出 GEMM。

## 四条证据，逐条排除"配置问题"

### ① 量化白名单**有** `fp8` —— 所以不是不认名字

```
supported_quantization = ['awq', 'awq_marlin', 'gptq', 'gptq_marlin', 'fp8',
                          'compressed-tensors', 'fbgemm_fp8', 'gguf', 'quark',
                          'mxfp4', 'gpt_oss_mxfp4', 'torchao', 'bitsandbytes',
                          'modelopt_fp4']
```

注册表也能解析：

```
fp8              -> Fp8Config
fp8_per_tensor   -> OnlineQuantizationConfig
awq              -> AWQConfig
gptq             -> GPTQConfig
compressed-tensors -> CompressedTensorsConfig
mxfp4            -> Mxfp4Config
```

（`fp8_e4m3` / `fp8_e5m2` / `fp8_w8a8` / `int8` / `w4a8` 报 `Invalid quantization method`
是正常的——这些不是合法的 `--quantization` 取值，合法名字是 `fp8`。）

### ② `fp8_dtype` **有**值 —— 所以不是 dtype 未定义

```
fp8_dtype = torch.float8_e4m3fn
```

### ③ 平台对象**缺** fp8 判定方法

```
is_fp8_supported:            MISSING
get_supported_attn_backends: MISSING
supported_attention_backends: MISSING
get_device_capability = DeviceCapability(major=9, minor=3)
```

`RocmPlatform` 继承了 `Platform` 基类的 `is_fp8_supported`（它用 `get_device_capability`
按 CUDA 9.x 推断），而这里的 `major=9, minor=3` 是从 **gfx936 硬套 CUDA 的 9.3** 来的
（gotchas / kernel-gate 已记：gfx936 被当作 CUDA 9.3 等价物）。
所以 vLLM 会在**启动时认为"fp8 应该支持"**，然后在真正做 GEMM 时撞上 ① 里的 `NotImplementedError`。
**这正好解释了 gotchas #22 里"`--quantization fp8` 在 engine core 初始化即失败"**——
失败发生在第一次预热 GEMM，看起来像初始化失败。

### ④ hipBLASLt **有** fp8 符号，但够不着

```
/opt/dtk/lib/libhipblaslt.so.0.10   (7.3 MB, 2026-03-17)
strings 里含  R_8F_E4M3   R_8F_E5M2
```

库认得 fp8 的 datatype 枚举，但**PyTorch 没有把它接进 `addmm` 的分发**。
这就是"库有、kernel 没有"的典型形态：需要一段 C++/HIP 绑定把 fp8 GEMM 暴露给 torch，
或让 vLLM 走一条不经过 torch GEMM 的专用路径（aiter 的 fp8 kernel 就是这种）。

## 与 aiter 的关系（这条把 P0-A 和 P0-B 串起来了）

aiter 里确实有 fp8 GEMM kernel，但：

- `is_aiter_found_and_supported()` 末行 `return on_mi3xx()`，
  `_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942","gfx950"])`；
- BW1000 是 `gfx936` → **恒 False**（job 1123605）；
- 环境变量 `VLLM_ROCM_USE_AITER_FP8BMM: bool = True` 默认开着，但被 `@if_aiter_supported` 短路。

**所以 P0-B 的结论和 P0-A 是同一个根因的两面**：
aiter 在 gfx936 上恒关 → fp8 GEMM 没有可用实现 → `--quantization fp8` 必失败。

**唯一的共同出口**是 job 1123927 发现的那条：`RocmPlatform.get_attn_backend_cls()`
的显式 `--attention-backend` 分支**跳过** `_get_backend_priorities`。
但它只解决 attention，**不解决 GEMM**——GEMM 没有等价的"显式指定"旋钮。

## 对 HANDOFF §3.2 B 的回答

| 原假设 | 判定 |
| --- | --- |
| "缺 hipBLASLt" | ❌ 库在，7.3 MB，含 fp8 符号 |
| "白名单没开" | ❌ `fp8` 在 `supported_quantization` 里，`Fp8Config` 能解析 |
| "要补 aiter" | ⚠️ 方向对，但补不了：`on_mi3xx()` 硬件白名单 |
| **"DCU 上无 fp8 GEMM kernel"** | ✅ **就是这条** |

**建议**：在拿到 `torch.addmm` 支持 fp8（或 vLLM 提供绕过 torch 的 fp8 GEMM 路径）
之前，**不要做任何自量化**（AWQ 那条线也一样，见 `docs/quantization-branch.md`：
int4→bf16 dequant→bf16 GEMM，慢 5×）。这条应从"待办"降级为"阻塞项"，
写进 `docs/gotchas.md`，不要再占卡时。

## 复现

```bash
sbatch sweeps/kimi-quantigate-2026-11-15.slurm
# 0.21（HCUPlatform）另需 QG_021=1 环境变量，但 0.21 对 Kimi 本身不可用，优先级低
```

## 遗留

- **0.21 的 `vllm_hcu` 有独立的量化白名单**，本次只跑了 0.20。
  0.21 用的是 `HCUPlatform`（非 `RocmPlatform`），可能自己实现了 fp8 GEMM 绑定。
  但 0.21 对 Kimi 不可用（3D conv1d 权重 `.t()` 崩），且 fp8 是给 Qwen3-ASR 用的，
  所以这条**值得单独确认一次**，属于 P0-B 的剩余工作。
