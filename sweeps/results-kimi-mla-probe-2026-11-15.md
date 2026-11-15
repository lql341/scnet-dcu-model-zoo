# Kimi MLA backend 溯源（0 卡时，2026-11-15）

> 作业：`kimi-bkprobe` 1122234、`kimi-mlaprobe` 1122316（各 1 卡只读，不加载模型）。
> 目的：16K/c64 崩溃时 `--attention-backend` 到底能不能换、能换成什么。

## 起点：一个被日志推翻的结论

排查初期我从 `kimi_linear.py` 的层构造点推断"`--attention-backend` 对 Kimi 完全无效"，
并写进了 README。**同日被 0.20 的实际启动日志推翻**（job 1120618）：

```
[rocm.py:547] Using TRITON_MLA backend out of potential backends: ['TRITON_MLA'].
```

Kimi 每层是 KDA / MLA 二选一：**KDA 绕开选择器，MLA 走选择器**。所以这条旋钮对一半的层有效。

## 为什么候选集里只有一个 TRITON_MLA

`vllm/platforms/rocm.py:361-383` 的 `_get_backend_priorities`：

```python
if use_mla:
    if rocm_aiter_ops.is_mla_enabled():
        return [ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]
    else:
        return [TRITON_MLA]          # ← 实测落在这里
```

`is_mla_enabled` 带 `@if_aiter_supported` 装饰器（1122316 抓到源码），aiter 不可用时整个函数被短路。
运行期实测：

```
is_aiter_found_and_supported = False
is_enabled / is_mla_enabled / is_mha_enabled / ... = None   （被装饰器短路）
```

## 因果链（两张探测拼起来的）

| 环节 | 证据 |
| --- | --- |
| 0.20 容器里 `import aiter` 失败 | 闸门 1121990：`Get GPU arch from rocminfo failed ... 'rocminfo'`（容器内无该命令） |
| → `is_aiter_found_and_supported()` 返回 False | 1122316 运行期实测 |
| → `is_mla_enabled()` 被 `@if_aiter_supported` 短路为 None | 1122316 源码 + 实测 |
| → `_get_backend_priorities(use_mla=True)` 只返回 `[TRITON_MLA]` | 1122316 源码 |
| → Kimi 的 MLA 层只能用 Triton MLA kernel | 1120618 启动日志 |

**结论**：0.20 上"MLA 只有 TRITON_MLA 可选"**不是 vLLM 的设计选择，而是一个环境缺口的下游后果**——
容器缺 `rocminfo`，aiter 初始化不了，aiter MLA 全部被关掉。

## 由这条链推出的具体动作

1. **补 `rocminfo`**（或让 aiter 的 arch 探测走别路）：可能让 `ROCM_AITER_MLA` 出现在候选集里。
   这比改 kernel 便宜得多，且不动权重。
2. **显式指定 `--attention-backend FLASHMLA`**：需先确认 selector 是否允许绕过候选集
   （探测 1122426 正在查）。
3. 若两者都不成，16K/c64 只剩 `--enforce-eager` / `--max-model-len` 这类配置绕行（作业 1122422）。

## 与 0.21 的关系

0.21 的 aiter **是可用的**（闸门实测导出 40+ 符号），所以它的 MLA 候选集本应更宽。
但 0.21 对 Kimi 有另一个更早的硬阻塞（`vllm_hcu` 用 `.t()` 加载 3D conv1d 权重），
所以"0.21 的 aiter 更好"这条优势**用不上**。见 `models/...__vllm/README.md` §7.3。

## ⚠️ 下面这段结论**已作废**（探测本身有 bug，见本节末尾）

## 一度以为的"决定性结果"：所有 MLA backend 在 DCU 上都不可用（作业 1122565）

把 `AttentionSelectorConfig(use_mla=True, head_size=128, dtype=bfloat16, block_size=16)` 直接喂给
每个 MLA backend 的 `validate_configuration()`，结果：

```
device_capability = DeviceCapability(major=9, minor=3)

ROCM_AITER_MLA          ['dtype not supported']
ROCM_AITER_TRITON_MLA   ['dtype not supported']
ROCM_AITER_MLA_SPARSE   ['dtype not supported', 'non-sparse not supported']
TRITON_MLA              ['dtype not supported']
CUTLASS_MLA             ['head_size / dtype / block_size / compute capability not supported']
FLASHMLA                ['head_size / dtype / block_size not supported',
                         'vllm._flashmla_C is not available, likely was not compiled ...']
FLASHINFER_MLA          ModuleNotFoundError: flashinfer
FLASH_ATTN_MLA          ImportError: 需 CUDA flash attention 扩展
```

**判读**：

1. **连当前实际在用的 `TRITON_MLA` 都报 `dtype not supported`**。这说明启动日志里
   `Using TRITON_MLA backend` 是**没有经过 `validate_configuration` 的那条单候选路径**
   （`_get_backend_priorities` 的 else 分支直接返回，不做筛选）。
2. `FLASHMLA` 明确报 `vllm._flashmla_C is not available` —— **该扩展在 DCU 镜像里根本没编译**，
   与"候选集没列出来"是两回事：即使强行指定也起不来。
3. 因此**"换 attention backend"对 Kimi 的 MLA 那一半也不成立**。上一版说的
   "MLA 层经过选择器，所以能换" 只说对了机制、说错了可行性：唯一的候选就是那个
   正在崩溃的 Triton MLA。

**结论修正（最终）**：16K/c64 在 0.20 上**没有任何 backend 可换**。剩下只有：
`--enforce-eager`、`--max-model-len`、`--max-num-seqs` 这三类配置绕行，
或承认这是 kernel 缺陷、把 16K/c64 记为不支持。

### 作废原因（作业 1122747 抓到，同日下午纠正）

1122747 打出了各 backend 的实际声明：

```
TRITON_MLA   supported_dtypes = [torch.float16, torch.bfloat16]
FLASHMLA     supported_dtypes = [torch.float16, torch.bfloat16]
```

而我在 1122565 里传的是**字符串** `dtype='bfloat16'`，`supports_dtype()` 拿它和 `torch.bfloat16`
比较，永远不相等 → **所有 backend 无条件报 `dtype not supported`**。
这不是镜像的缺陷，是我的探测写错了类型。

**教训**：`validate_configuration` 的 `dtype` 参数要的是 `torch.dtype` 对象，不是字符串。
用"全表都返回同一个理由"这种模式判读结论时要立刻怀疑自己的入参——真实世界里不会所有实现
连失败原因都一模一样。

已用 `torch.bfloat16` 重测（作业 1122898），结果见下节。

## 修正后的真实结果：aiter MLA 两兄弟可用，FLASHMLA 不可用（作业 1122898）

用 `dtype=torch.bfloat16`、`head_size` 分别取 128 / 192 / 72 各测一遍，三个取值结论一致：

| backend | 128 | 192 | 72 | 说明 |
| --- | --- | --- | --- | --- |
| **`ROCM_AITER_MLA`** | ✅ OK | ✅ OK | ✅ OK | **可换目标** |
| **`ROCM_AITER_TRITON_MLA`** | ✅ OK | ✅ OK | ✅ OK | **可换目标** |
| `TRITON_MLA` | ✅ OK | ✅ OK | ✅ OK | 当前在用（崩溃的那个） |
| `ROCM_AITER_MLA_SPARSE` | ✗ non-sparse | ✗ | ✗ | 只在 sparse 路径 |
| `CUTLASS_MLA` | ✗ head_size/block_size/cc | ✗ | ✗ | 不适用 |
| `FLASHMLA` | ✗ head_size/block_size + **`_flashmla_C` 未编译** | ✗ | ✗ | **不可用** |
| `FLASHMLA_SPARSE` | ✗ | ✗ | ✗ | 只在 sparse |
| `FLASHINFER_MLA*` | ✗ 无 flashinfer 包 | ✗ | ✗ | 不可用 |
| `FLASH_ATTN_MLA` | ✗ 需 CUDA FA 扩展 | ✗ | ✗ | 不可用 |

## 由此确定的下一步（终于有方向了）

**`ROCM_AITER_MLA` 可用，但它需要 aiter 初始化成功**，而 0.20 容器缺 `rocminfo` 导致
`is_aiter_found_and_supported()=False`，所以它进不了候选集。

→ **动作：给 0.20 容器补 `rocminfo`，然后要么靠优先级自动选中 `ROCM_AITER_MLA`，
要么显式 `--attention-backend ROCM_AITER_MLA` 跑 16K/c64。**

这条路径的两个前提都已单独验证过：
1. `validate_configuration()` 说它 OK（本作业）；
2. 它的 kernel（`chunk_gated_delta_rule_fwd` 等）在 0.21 的 aiter 里确实存在（闸门 1121990）。

### 实测复核（作业 1123060）：只补 `rocminfo` **不够**

这次探测推翻了我上面给自己写的两个假设 → 结论要改。

**假设 1「容器里没有 `rocminfo`」错了。** 它在，只是不在 `PATH` 上：

```
FOUND /opt/dtk/bin/rocminfo
missing /opt/dtk/.hyhal/rocm_smi/bin/rocminfo
missing /usr/bin/rocminfo
PATH 里没有 rocminfo
```

**假设 2「补上就能让 aiter 可用」也错了。** 用一个假 `rocminfo` 塞进 `PATH` 后：

```
aiter OK with fake rocminfo        ← import 成功了
is_aiter_found_and_supported = False
is_mla_enabled = None
priorities(use_mla=True) = ['TRITON_MLA']
```

也就是说 `import aiter` 只是**第一道门**。`is_aiter_found_and_supported()` 过了这一关之后
还有自己的判定（`_aiter_ops.py:42`），而它在这台机器上仍然为 False。所以
"缺 `rocminfo` → aiter 关掉"只是**半个根因**；后面还有第二个开关没找到。

**仍然成立的部分**：

- `/opt/dtk/bin/rocminfo` 存在 → 只要把它加进容器的 `PATH`，`import aiter` 这一关就能过，
  不需要伪造二进制（这是修法，不是绕过）。
- `torch.cuda.get_arch_list()` 为 `['gfx906','gfx926','gfx928','gfx936','gfx938']`，
  `get_device_name(0)` = `BW` —— arch 探测的原料本身是齐的。

**新增动作**：查 `_aiter_ops.py:42` 的 `is_aiter_found_and_supported()` 与
`rocm_aiter_ops.is_mla_enabled()` 的判定条件（作业 1123259）。这一枪不花卡时。

### 第二道门找到了：`on_mi3xx()`（作业 1123513，源码全文）

`_aiter_ops.py` 里的实现只有 5 行有效逻辑：

```python
42: def is_aiter_found_and_supported() -> bool:
58:     if current_platform.is_rocm() and IS_AITER_FOUND:
59:         from vllm.platforms.rocm import on_mi3xx
61:         return on_mi3xx()          # ← 决定性的一行
62:     return False
```

也就是说：**aiter 能不能用，最终取决于"这块卡是不是 MI300 系"**。
`IS_AITER_FOUND` 只是 `find_spec("aiter") is not None`（包在不在），
过了这一关之后还要 `on_mi3xx()` 为真。

BW1000 是 `gfx936`，不是 MI300（`gfx942`）。所以：

- **补 `rocminfo` 解决的是 `import aiter`**（第一道门，供 `IS_AITER_FOUND` 之前的 JIT 探测用）；
- **`on_mi3xx()` 是第二道门，与 rocminfo 无关**，在 BW1000 上恒为假。

**结论**：`_get_backend_priorities(use_mla=True)` 在 0.20 上返回 `[TRITON_MLA]`
是**上游对 MI300 的硬编码判断**，不是环境缺口。补 `rocminfo` 或设 `PATH` 都不会改变它。
**除非**显式 `--attention-backend ROCM_AITER_MLA` 绕过候选集
（该函数 docstring 明写："Backend selection: Can explicitly request aiter regardless of env var"）。
→ **这一条已被 1123927 证实**：`rocm.py:478` 的显式分支确实不调用 `_get_backend_priorities`。
剩下只需验证 `get_class()` 本身能否加载（作业 1124018）。

这条是上一个假设的**第二次修正**。第一次以为"缺 rocminfo"，第二次以为"PATH 不对"，
现在是：**两道门，第二道是硬件型号白名单。**

### `on_mi3xx()` 的判据：一行硬编码（作业 1123605）

```python
187: _ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
189: _ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])   # ← 白名单
190: _ON_GFX9  = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
275: def on_mi3xx() -> bool:
276:     return _ON_MI3XX
```

BW1000 是 **gfx936**，不在 `["gfx942","gfx950"]` 里 → `_ON_MI3XX = False`，
与 `rocminfo`、`PATH`、环境变量**全都无关**。

这也解释了一个矛盾现象：0.21 的 aiter 明明"能 import 且导出 40+ 符号"（闸门 1121990），
但那不等于 0.21 会启用 aiter MLA —— 同一套 `on_mi3xx()` 判定在那边同样为假。
（0.21 用的是 `HCUPlatform`，是否另有一套判定，需要用 `vllm_hcu` 的源码单独确认。）

## 本日排查路径（五次探测、两次自我纠错）

| # | 作业 | 目的 | 结论 |
| --- | --- | --- | --- |
| 1 | 1121990 | 闸门：两镜像的能力面 | 0.20 有 aiter 但 `import` 失败（缺 `rocminfo`）；0.21 的 aiter 可用 |
| 2 | 1122012 | 0.21 跑 Kimi（4 卡） | ❌ 加载权重即崩：`vllm_hcu` 对 3D conv1d 权重调 `.t()` |
| 3 | 1122234 | 0.20 的 MLA backend 判定逻辑 | 候选集由 `_get_backend_priorities` 决定；`is_mla_enabled()` 为假 → 只有 `TRITON_MLA` |
| 4 | 1122316 | aiter 运行期状态 | `is_aiter_found_and_supported()=False`；`@if_aiter_supported` 把一组开关全短路 |
| 5 | 1122565 | 显式指定各 MLA backend 是否可用 | 初步结论"全不可用" → **作废**（dtype 传了字符串） |
| 6 | 1122747 | 查 `supported_dtypes` 与 `validate_configuration` 源码 | 发现上一条的 bug；`FLASHMLA` 的 `_flashmla_C` 确实未编译 |
| 7 | 1122883/1122898 | 用 `torch.bfloat16` 重测 | 见下节 |
| 8 | 1123060 | 补 `rocminfo` 能否救活 aiter | 二进制在 `/opt/dtk/bin`（不在 PATH）；补上后 `import` 成功但 `is_aiter_found_and_supported()` 仍 False → 还有第二道门 |
| 9 | 1123259 | 查第二道门的判定条件 | `is_aiter_found_and_supported()` 末行 `return on_mi3xx()` |
| 10 | 1123605 | `on_mi3xx()` 的判据 | `_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942","gfx950"])`，gfx936 恒 False |
| 11 | 1123697 | 运行期复核 | `on_mi3xx()`=False、平台 `RocmPlatform`、`capability=(9,3)` |
| 12 | 1123796 | 显式 `--attention-backend` 走哪条路径 | **`selector.py` 根本没有 `attention_backend` 参数** → 显式指定不经选择器，需另找消费点（进行中 1123917） |

### ⚠️ 上面"显式指定也绕不过"的说法**不准确**——它确实能绕过（作业 1123927，`rocm.py:468-497` 全文）

`RocmPlatform.get_attn_backend_cls()` 的第一段就是显式分支，**完全不碰
`_get_backend_priorities` / `on_mi3xx()`**：

```python
468:    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None) -> str:
474:        device_capability = cls.get_device_capability()
477:        # First try checking just the selected backend, if there is one.
478:        if selected_backend is not None:
479:            try:
480:                backend_class = selected_backend.get_class()
481:                invalid_reasons = backend_class.validate_configuration(
482:                    device_capability=device_capability,
483:                    **attn_selector_config._asdict(),
484:                )
485:            except ImportError:
486:                invalid_reasons = ["ImportError"]
487:            if invalid_reasons:
488:                raise ValueError(f"Selected backend {selected_backend} is not valid ...")
489:            else:
493:                logger.info_once("Using %s backend (selected via --attention-backend).", ...)
497:                return selected_backend.get_path()
501:        valid_backends_priorities, invalid_reasons = cls.get_valid_backends(...)   # ← 只有这条路才查 on_mi3xx
```

| 路径 | 是否经过 `_get_backend_priorities` | 是否受 `on_mi3xx()` 限制 |
| --- | --- | --- |
| 默认（不传 `--attention-backend`） | 是（`rocm.py:501`） | **是** |
| 显式 `--attention-backend X` | **否**（`rocm.py:478`） | **否** |

所以 `--attention-backend ROCM_AITER_MLA` **不会被 `on_mi3xx()` 拦**。
它只过 `validate_configuration()`，而 1122898 实测该方法对 `ROCM_AITER_MLA` 返回空列表（OK）。

**且这个关卡也已验证通过（作业 1124018）**。先 `export PATH=/opt/dtk/bin:$PATH`
（让 `import aiter` 能过第一道门），然后：

```
PATH 含 /opt/dtk/bin? NO        ← 容器默认确实没有
/opt/dtk/bin/rocminfo           ← 补上后可用
aiter OK

IS_AITER_FOUND = True
_GCN_ARCH      = gfx936
_ON_MI3XX      = False          ← 第二道门依然关着
is_aiter_found_and_supported() = False
is_mla_enabled() = None

get_path()  = vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend
get_class() = <class '...AiterMLABackend'>          ← ✅ 成功，无 ImportError
supports_mla = ['auto','float16','bfloat16','fp8','fp8_e4m3','fp8_e5m2']
validate_configuration -> OK (empty)                ← ✅ 通过
```

**关键**：`is_aiter_found_and_supported()` 仍是 **False**、`is_mla_enabled()` 仍是 **None**，
但 `get_class()` 和 `validate_configuration()` **都成功了**——因为
显式分支根本不调用它们（`rocm.py:478` 那一行）。

→ **结论：`--attention-backend ROCM_AITER_MLA` 在 BW1000 上可用。**
这是唯一一条绕过 `on_mi3xx()` 硬件白名单的路径。
实际跑 16K/c64 的作业是 **1124165**，结果见 `results-kimi-16k-c64-2026-11-15.md`。

**注意**：这条只解决 **attention**。GEMM（fp8）没有等价的"显式指定"旋钮，
所以 P0-B 仍然阻塞（见 `results-kimi-quantigate-2026-11-15.md`）。

**另有两条旁证**（`rocm.py` `_get_backend_priorities` 非 MLA 分支）：

```
383:    if is_aiter_found_and_supported():          ← 同一道 on_mi3xx 门
384:        backends.append(AttentionBackendEnum.ROCM_AITER_UNIFIED_ATTN)
```

即**不只是 MLA**，所有 aiter backend 在 BW1000 上都被这一行挡在候选集之外。
显式指定是唯一入口，这条结论对 P0-B（fp8 GEMM）同样适用。

### `@if_aiter_supported` 全文（作业 1123796，源码）

```python
88: def if_aiter_supported(func: Callable) -> Callable:
89:     """Decorator that only executes the function if
90:         ROCm AITER package is supported and enabled on gfx9 archs.
91:     """
93:     @functools.wraps(func)
94:     def wrapper(*args, **kwargs):
95:         if is_aiter_found_and_supported():
96:             return func(*args, **kwargs)
98:         return None
```

**注意**：装饰器只拦"函数调用"，不拦"枚举值的存在"。也就是说
`AttentionBackendEnum.ROCM_AITER_MLA` 这个枚举在 DCU 上始终可解析，
真正拦人的是 `is_aiter_found_and_supported()`，而它末行是 `on_mi3xx()`。
所以 **任何依赖该装饰器的旋钮（含 `is_mla_enabled()`）在 gfx936 上都没救**；
唯一可能绕开的路径是"某处直接按用户显式指定的字符串 import backend 类"，
而不是走 `_get_backend_priorities` 的候选集。详见作业 1123917。

### 一个与主线无关、但影响所有压测结论的 bug（2026-11-15 修掉）

`bench.sh` 的有效性校验段：

```python
import json, sys                       # ← 漏了 os
...
ROUND_FILE = "bench/round2.jsonl" if os.path.exists(...) else ...
```

`os` 未导入 → `NameError` → **这段"TTFT=0 即判定无效"的闸门从未执行过**。
16K/c64 的 B 档（全部 400、`e2e_latencies` 为空）本该在这里被拦住并报
`INVALID ... TTFT=0`，实际却是下游 `np.percentile` 抛 `IndexError`，
把"配置错误"表现成了"脚本崩溃"。

同一段模板被复制到全部 10 个 `bench.sh`，已一并修复（`import json, os, sys`）。
**教训**：有效性校验自己失效时，它保护的那些结论全部要重新审视。

**两次自我纠错都写进 gotchas #42**（"全表返回同一理由"要怀疑入参）。
今天唯一一条被三次改写的结论是 §7.2，最终形态是：
**机制上 MLA 层走选择器，但当前候选集只有 `TRITON_MLA`，根因是容器缺 `rocminfo`。**
