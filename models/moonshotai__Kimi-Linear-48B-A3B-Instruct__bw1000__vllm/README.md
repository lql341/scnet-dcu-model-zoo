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

早期排查记录过「vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio），
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

取 `bench/round2.jsonl`（单位 output tok/s），与同模型 SGLang 条目并列。
⚠️ **16K/c64 那行是例外**：该档不能"连跑两轮"取 round2（第二轮必崩），
数字来自 job 1124366 的"每轮重启服务"实验（原始数据在 `bench/restart-1124366/`），
表中填稳态三轮的中位数（R3 = 284.62）。

| 档位 | **vLLM 0.20** | SGLang 0.5.10 | vLLM 相对 |
| --- | ---: | ---: | ---: |
| 4K / c1 | 40.73 | 104.67 | −61% |
| 4K / c8 | 304.18 | 353.53 | −14% |
| 4K / c64 | **858.08** | 839.00 | **+2%** |
| 16K / c1 | 37.36 | 50.80 | −26% |
| 16K / c8 | 197.83 | 119.21 | **+66%** |
| 16K / c64 | **283.53–286.84**（4 轮实测，每轮重启服务；⚠️ 见下） | 234.17 | **+22%** |

> **16K/c64 的前提：每轮压测前必须重启服务进程。** 原因是累计状态问题（见下），
> 不是配置问题。冷启动约 3–4 分钟，换来可重复的有效数据。
>
> **三条绕行路线的结果**（2026-11-15 下午）：
> | 路线 | 作业 | 结果 |
> | --- | --- | --- |
> | `--max-num-seqs` 24 / 16 / 32+eager | 1123874 | 三档 round1 都过，**round2 全崩**；TTFT 恶化到 156–188 s |
> | `--attention-backend ROCM_AITER_MLA` | 1124165 | ❌ **服务起不来**：`ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'` |
> | 同上 + `--enforce-eager` | 1124165 | ❌ 同一个 import 路径，同样起不来 |
> | **每轮重启服务** | **1124366** | ✅ **4/4 轮全过**：R2–R4 稳定在 283.53 / 284.62 / **286.84 tok/s**，各 128/128、0 处 VMFault（R1 235.14 为冷启动，不作稳态引用） |
>
> 详见 `sweeps/results-kimi-16k-c64-2026-11-15.md`、gotchas #49/#50。

**结论（与 MiMo 的"vLLM 全面落后"不同，值得记录）**：
- **高并发档 vLLM 反超**：4K/c64 +2%、16K/c8 **+66%**。Kimi 是 48B 小激活（3B）MoE-like 结构，
  vLLM 的调度在高并发下更吃得开。
- **单流 vLLM 明显落后**（4K/c1 −61%），说明低并发时 SGLang 的 kernel 更优。
- **16K/c64 的崩溃是「累计状态」问题，不是配置问题（2026-11-15 定论，job 1124366）**。
  五个配置（默认 / eager / `--max-num-seqs` 24 / 16 / 32+eager）全部"第一轮过、第二轮崩"，
  崩溃点分散在 `Running=` 29 / 62 / 23 / 16 —— **与并发数、cudagraph、批大小都无关**。
  重读 job 1120618 的日志才看清：那次单进程连跑 6 档，**运行 31 分钟、约 1200 请求后**
  才在 16K/c64 的首轮崩掉。→ 变量是**服务进程的累计运行时长/处理量**。

  **解法：每轮压测前重启服务进程。** 实测**连续四轮各 128/128、0 处 VMFault**，
  稳态吞吐 **283.53–286.84 tok/s**（R2–R4 极差 1.2%）、TTFT **30.6–31.5 s**，
  且不需要任何降级开关。首轮 235.14 tok/s（TTFT 47.5 s）是页缓存未暖，不作稳态引用。
  代价是每轮多 3–4 分钟冷启动。详见 `sweeps/results-kimi-16k-c64-2026-11-15.md`、
  gotchas #51。
  **注意单轮成功不能当结论**——本条目曾据第 1 轮数据误记"已跑通"，被第 2 轮推翻；
  而"第二轮必崩"也曾被误读成"批大小问题"，浪费了三个作业。

  原始崩溃证据（默认配置，graph 开）：
  不是"客户端没起来"，而是**服务端 worker 被 GPU 内存错误打死**：

  ```
  Invalid address access: 0x7f7e01015000, Error code: 1.
  >>>>>>>> KERNEL VMFault !!!! <<<<<<
  (EngineCore) ERROR [multiproc_executor.py:283] Worker proc VllmWorker-3 died unexpectedly
  (APIServer) ERROR [serving.py:448] EngineDeadError: EngineCore encountered an issue
  ```

  崩溃时调度器状态 `num_running_reqs=33, num_waiting_reqs=31`（16K 输入 × c64）。
  同配置下 **16K/c1、16K/c8 都正常**（本轮实测 34.78 / 184.51 tok/s），
  所以这是**大 batch + 长上下文下的 kernel 地址越界**，与显存容量无关（无 OOM 记录）。
  上一版把根因写成"缺 paged-attn 内核"是推测；现在有 VMFault 的原始证据，
  结论收紧为：vLLM 0.20 在 DCU 上跑 16K×c64 会触发 GPU 访存错误，属框架/kernel 缺陷，非配置问题。

  排查过程中另修掉两个**会掩盖真因**的问题：
  1. 压测客户端 `--ready-check-timeout-sec` 默认 60 s，而本模型冷启动要 410 s，
     于是"服务还没起来"被记成"该档不支持"。已改为可配（默认 900 s），否则永远看不到 VMFault。
  2. **`bench.sh` 的有效性校验从未执行过**（`import json, sys` 漏了 `os` → `NameError`，
     校验段在读到第一个文件前就死了）。于是"全部 400 + TTFT=0"这类无效数据
     不但能进 `round2.jsonl`，还会以 `np.percentile` 的 `IndexError` 收场，
     把"配置错误"伪装成"脚本崩溃"。2026-11-15 已修全部 10 个 `bench.sh`（gotchas #45）。
     **在此之前本仓库所有 `bench.sh` 的"all valid"都没真正打印过。**

### 7.1 多卡起服务必须显式给可见性

Slurm 只把 1 张卡写进 `CUDA_VISIBLE_DEVICES`，容器默认继承后 vLLM 只看到 1 张卡，
TP=4 直接 `Engine core initialization failed`。必须：

```bash
podman run ... -e HIP_VISIBLE_DEVICES=0,1,2,3 -e CUDA_VISIBLE_DEVICES=0,1,2,3
```

这条已写入 `container/submit.slurm`，也是 gotchas #21 的补充。

### 7.2 `--attention-backend` 对 Kimi：机制上走选择器，但**实际无可选**（2026-11-15 三次探测后定论）

> 这一节改过两次，最终结论如下。
> 1. 初版："完全无效" —— **机制上说错了**（MLA 层确实走选择器）。
> 2. 二版："MLA 层能换" —— 机制对，但可行性未验证，且候选集只有一项。
> 3. 现版：机制成立；候选集只有 `TRITON_MLA`，原因见下（缺 `rocminfo` → aiter 被关）。
>    其它实现（`FLASHMLA` / `FLASHINFER_MLA`）因扩展未编译而不可用；aiter MLA 能否启用待 1122883。

`kimi_linear.py` 里每层二选一：

```
21:   from vllm.model_executor.layers.kda import KimiDeltaAttention
177:  class KimiMLAAttention(nn.Module)
302:      if config.is_kda_layer(layer_idx):
303:          self.self_attn = KimiDeltaAttention(...)     # 线性注意力（KDA）
312:      else:
312:          self.self_attn = KimiMLAAttention(...)       # MLA
618/634/648:  MambaState{Dtype,Shape,CopyFunc}Calculator.kda_state_*()
```

- **KDA 层**（`KimiDeltaAttention`）不是 `AttentionBackend` 实现，不经选择器 → `--attention-backend` 管不到。
- **MLA 层**（`KimiMLAAttention`）是普通注意力，**会**经选择器。实测日志（0.20，job 1120618）：
  ```
  [rocm.py:547] Using TRITON_MLA backend out of potential backends: ['TRITON_MLA'].
  ```
  即候选集只有 `TRITON_MLA` 一个。同一日志还有：
  ```
  [compilation.py:1370] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with
    TritonMLABackend backend (support: AttentionCGSupport.NEVER); setting cudagraph_mode=PIECEWISE
  ```

因此"换 backend"仍是一条真实可走的路（至少对 MLA 那一半），只是要先让 `RocmPlatform` 的候选集里
出现 `FLASHMLA` 等其它实现。

**候选集的判定逻辑（`vllm/platforms/rocm.py:361-383`，0.20 只读探测）**：

```python
from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
...
    if rocm_aiter_ops.is_mla_enabled():
        return [ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]   # 364-371
    else:
        return [TRITON_MLA]                                          # 375
```

实测日志落在 **else 分支**（只有 `TRITON_MLA`），即 `rocm_aiter_ops.is_mla_enabled()` 为 False。
而闸门探测已证实：**0.20 容器的 `import aiter` 本身就失败**
（`Get GPU arch from rocminfo failed ... 'rocminfo'`，容器内无该命令）。
→ **两条事实是同一条因果链**：缺 `rocminfo` → aiter 不可用 → aiter MLA 被禁用 → 只剩 `TRITON_MLA`。

这条给了第一个具体动作：**补 `rocminfo`**，让 aiter 初始化，可能让 `ROCM_AITER_MLA` 出现在候选集里。

**一个已证伪的中间结论（记下来防重犯）**：曾用 1122565 得出"所有 MLA backend 都报
`dtype not supported`、故无可换"的结论，但那是**探测脚本传了字符串 `'bfloat16'` 而非
`torch.bfloat16`** 造成的假象——`supports_dtype()` 拿字符串和 `torch.dtype` 比较永远为假。
已用正确的 `torch.dtype` 重测（作业 1122883），结论以重测为准。

**可确定的一点**：`FLASHMLA` 报 `vllm._flashmla_C is not available`，
该 C++ 扩展在 DCU 镜像里**没有编译**；`FLASHINFER_MLA` 缺 `flashinfer` 包。
所以即使候选集放宽，这两个也用不了。
完整记录见 `sweeps/results-kimi-mla-probe-2026-11-15.md`。

**最终定论（2026-11-15，五轮探测后）——候选集在 BW1000 上不可能变宽**：

aiter 被**两道门**挡住，第二道是硬件型号白名单：

| 门 | 判据 | BW1000（gfx936） |
| --- | --- | --- |
| 1 | `import aiter` 需要 `rocminfo` 能报 arch | 二进制在 `/opt/dtk/bin`，**不在 PATH**（可修） |
| 2 | `is_aiter_found_and_supported()` 末行 `return on_mi3xx()` | `_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942","gfx950"])` → **恒 False** |

所以 `rocm_aiter_ops.is_mla_enabled()` 被 `@if_aiter_supported` 永久短路成 `None`
→ `_get_backend_priorities(use_mla=True)` 永久返回 `[TRITON_MLA]`。
**补 `rocminfo`/改 PATH 全部无效**，这是上游 vLLM 对 MI300 的硬编码，不是环境缺口。

`@if_aiter_supported` 源码（job 1123796）：

```python
def if_aiter_supported(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_aiter_found_and_supported():
            return func(*args, **kwargs)
        return None
```

**但显式 `--attention-backend` 这条路是通的**（2026-11-15 最终确认）：

`RocmPlatform.get_attn_backend_cls()`（`rocm.py:468-497`）有两条路径：

```python
if selected_backend is not None:              # ← 478: 用户显式指定
    backend_class = selected_backend.get_class()
    invalid_reasons = backend_class.validate_configuration(...)   # 只查配置兼容性
    return selected_backend.get_path()        # 不碰 _get_backend_priorities / on_mi3xx

valid_backends_priorities, _ = cls.get_valid_backends(...)        # 501: 默认路径，才查白名单
```

实测（job 1124018，先补 `PATH=/opt/dtk/bin`）：

```
on_mi3xx() = False            is_mla_enabled() = None     ← 白名单依旧是关的
get_class() -> AiterMLABackend                              ← ✅
validate_configuration -> OK (empty)                        ← ✅
```

→ 这条只证明了**能绕过白名单、拿到类名**，但**不证明实现可用**。
job 1124165 实际起服务时，四个 rank 在引擎初始化阶段同时报：

```
ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'
(/usr/local/lib/python3.10/dist-packages/aiter/__init__.py)
```

`validate_configuration()` 不导入 backend 模块，真正的 import 发生在初始化时。
vLLM 0.20 的 `rocm_aiter_mla.py` 需要 aiter 导出 `flash_attn_varlen_func`，
而镜像自带的 aiter 版本没有这个符号——**vLLM 与 aiter 版本错配，与 gfx936 无关**。

→ **`--attention-backend ROCM_AITER_MLA` 在 BW1000 上不可用（服务起不来）**，
`--enforce-eager` 走同一 import 路径，同样失败（实测）。见 gotchas #49。

### 7.3 换镜像到 0.21 **不可行**：`vllm_hcu` 加载 3D 权重即崩（2026-11-15 实测）

闸门发现 0.21 自带 `causal_conv1d 1.5.4`，原判断"换镜像是最便宜的一枪"。
**实测否掉了这条路**（job 1122012，`--max-model-len 17408`、TP=4、`--max-num-seqs 64`）：
四个 rank 在**加载权重阶段**全部失败，服务根本没起来，压测一步都没跑到。

```
(Worker_TP0) File ".../vllm/model_executor/models/kimi_linear.py", line 565, in load_weights
                weight_loader(param, loaded_weight, **kwargs)
(Worker_TP0) File ".../vllm_hcu/model_executor/layers/linear.py", line 629, in weight_loader_v2
                param.load_column_parallel_weight(loaded_weight=..., is_quantization=...)
(Worker_TP0) File ".../vllm_hcu/model_executor/parameter.py", line 158, in load_column_parallel_weight
                loaded_weight = loaded_weight.t()
RuntimeError: t() expects a tensor with <= 2 dimensions, but self is 3D
```

判读：`vllm_hcu` 是海光自研的并行线性层实现，其 `load_column_parallel_weight` **无条件**对权重转置，
而 Kimi 的 `conv1d` 权重是 3D（`[dim, 1, kernel]`）。0.20 用的是 vLLM 上游的 `parameter.py`，
不会这样写，所以同一份权重在 0.20 上能加载。

**结论**：0.21 对 Kimi 是**比 0.20 更差**，不是更好。16K/c64 的排查必须留在 0.20 镜像内做，
可用的旋钮只剩 `--attention-backend`（MLA 一半）、`--max-model-len`、`--max-num-seqs`、`--enforce-eager`。

## 8. 注意事项

- 首次起服务会触发 vLLM 对 Kimi 线性注意力的 kernel 编译，冷启动可能 >10 min（`LOAD_TIMEOUT` 设 2400 s）
- 本机 NFS 上 `cp -rf` 会漏拷/不覆盖文件 → 同步一律用 `rsync`（gotchas #26）
- **`--attention-backend` 只对 MLA 那半数层有效**，见 §7.2；KDA 层绕开选择器
- **不要再用 0.21 镜像跑 Kimi**：`vllm_hcu` 加载 3D conv1d 权重即崩，见 §7.3
