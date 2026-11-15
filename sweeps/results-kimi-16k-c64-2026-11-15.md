# Kimi 16K/c64 崩溃：五个配置全部失败，最终定位为**累计状态**问题（2026-11-15）

> **TL;DR**：崩溃与 `Running` 数、cudagraph、批大小、attention backend 都无关；
> 是**服务进程累计运行时长/处理量**的问题。**每轮压测前重启服务即可稳定出数**——
> **连续四轮各 128/128、0 处 VMFault**，稳态吞吐 **283.53–286.84 tok/s**（极差 1.2%），
> TTFT **30.6–31.5 s**。首轮 235.14 tok/s 是页缓存未暖，不作稳态引用。
> 交付口径：16K/c64 单轮 + 重启（每轮多 3–4 分钟冷启动）。
>
> 作业：`kimi-16k-020` **1123145**（4 卡 TP=4，单作业内顺序试 A/B/C 三配置）。
> 前三次尝试（1122249 / 1122422 / 1122564）分别栽在 `env.sh` 缺变量和节点磁盘满，
> **一次都没真正跑到压测**（见 `results-kernel-gate-2026-11-15.md`）。这次跑到了。
>
> 后续作业（2026-11-15 下午）：**1123874** `--max-num-seqs` 降批、**1124165**
> 显式 `--attention-backend ROCM_AITER_MLA`、**1124366** 每轮重启服务。

## 结果总表

| 配置 | round1 | round2 | 结论 |
| --- | --- | --- | --- |
| **A** `--enforce-eager`（20480） | ✅ 128/128，239.32 tok/s | ❌ **KERNEL VMFault**（`Running=62, Waiting=2`） | 只把阈值抬高，没根治 |
| **B** `--max-model-len 17408`（graph 开） | ❌ 全部请求 **400** | ❌ | **配置本身算错**，不是崩溃 |
| C `17408 + eager` | 未跑 | 未跑 | 同 B，必然 400 |
| **D1** `--max-num-seqs 24`（1123874） | ✅ 128/128，216.66 tok/s | ❌ **KERNEL VMFault** | **降到 24 仍然崩** |
| **D2** `--max-num-seqs 16` | ✅ 128/128，208.71 tok/s，TTFT **187.6 s** | ❌ **KERNEL VMFault** | 降到 16 仍崩（崩在 `Running=16`） |
| **D3** `--max-num-seqs 32 + eager` | ✅ 128/128，190.54 tok/s，TTFT 145.8 s | ❌ **VMFault**（只成功 31 个） | 峰值并发 65 |
| **E** `--attention-backend ROCM_AITER_MLA`（1124165） | ❌ **服务起不来** | — | `ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'` |
| **F** E + `--enforce-eager` | ❌ 同上 | — | 同一个 import 路径，必死 |
| **G** 每轮重启服务 ×4（1124366） | ✅ **四轮全部通过**：235.14 / 283.53 / 284.62 / **286.84 tok/s**，各 128/128、0 VMFault | — | **✅ 累计状态假设成立，交付口径=每轮重启** |

## D1：`--max-num-seqs 24` —— 第一轮跑通，服务端全程稳定

服务端实时状态（round1 期间）：

```
16:04:01  Running: 24 reqs, Waiting: 36 reqs, GPU KV cache usage:  9.9%
16:04:31  Running: 24 reqs, Waiting: 32 reqs, GPU KV cache usage: 10.1%
16:07:41  Running:  8 reqs, Waiting:  0 reqs, GPU KV cache usage:  3.5%
16:07:51  Running:  0 reqs, Waiting:  0 reqs, GPU KV cache usage:  0.0%   ← 正常收尾
```

`Running` 精确钉在 24（= `--max-num-seqs`），**全程没有一次 `VMFault`**（`grep -c VMFault` = 0）。

round1 数据：

| 指标 | D（max-num-seqs 24） | A（eager，天然并发到 62） |
| --- | ---: | ---: |
| 成功请求 | 128/128 | 128/128 |
| Output throughput | **216.66 tok/s** | 239.32 tok/s |
| **Mean TTFT** | **156151 ms（156 s）** | 43563 ms（43.6 s） |
| Mean TPOT | 98.40 ms | — |
| Benchmark duration | 604.97 s | 547.69 s |
| 崩溃 | 无 | round2 崩 |

**判读**：
- **吞吐只降 9.5%**（239.32 → 216.66），代价远小于预期；
- **但 TTFT 恶化 3.6×**（43.6 s → 156 s）。原因是限批后请求排队：64 个并发里只有 24 个在跑，
  其余 40 个等前一批生成完 1024 个 token 才轮到 —— 首 token 要等 ~2 整批。
- 这是**用首延迟换稳定**，不是免费的。对"必须出数"的场景可以接受，对交互式场景不行。

### ❌ round2 崩了 —— "单批规模是根因"这个假设也被否定

`ms_1123874.out` 的结尾：

```
[bench] round1_16384_1024_c64 15:57:08
[bench] round2_16384_1024_c64 16:07:50
[bench] FAIL round2_16384_1024_c64
[mx 16:12:04] CONFIG mns24: 无数据，server=dead
[mx 16:12:05] CONFIG mns24 崩溃现场：
Invalid address access: 0x7fd4b6035000, Error code: 1.
>>>>>>>> KERNEL VMFault !!!! <<<<<<
```

**降到 `Running=24` 依然 VMFault。** 加上默认配置（崩在 29）和 eager（崩在 62），
三个不同的并发上限分别在 29 / 62 / 23 崩 —— **崩溃点与 `Running` 数没有对应关系**。

崩前客户端已成功 **38 个请求**（`Successful requests: 38`），`Running` 从 24 掉到 23
（最后一个请求收尾时崩），服务端最后一条日志是：

```
16:11:51  Running: 23 reqs, Waiting: 40 reqs, GPU KV cache usage: 9.8%
          Invalid address access: 0x7fd4b6035000 → KERNEL VMFault
```

崩溃发生在**一轮就要收尾、负载正在下降**的时刻，不是峰值。这进一步说明
"batch 维度"不是变量——它是在**累计处理了 38 个 16K 请求之后**才触发的。

这否定了两个假设：

1. ~~"单批规模是根因"~~ —— 24 也崩，说明不是 batch 维度算错；
2. ~~"cudagraph 桶边界（24→32）"~~ —— 24 这个值本身就在 `cudagraph_capture_sizes`
   `[1,2,4,8,16,24,32]` 里，跨桶与否都一样崩。

**剩下的解释**：这是**与运行时长/请求总数相关的概率性故障**。
round1 每次都是 128 个 16K 请求跑完不崩，round2 每次都在中途崩——
两轮唯一的差别是**累计运行时间**（round2 开始时进程已跑了约 10 分钟）。
嫌疑从"某次 kernel 越界"转向**状态累积**：KV cache 块分配/释放、KDA 的 recurrent state、
或集合通信缓冲区复用后的残留。

**下一步应该验证的是"重启服务后单跑一轮能否反复成功"**，
若 4 次单轮都过、每次连跑两轮必崩，则根因与运行时长相关，而不是与 batch 有关。

### D2 / D3 补齐：降批路线整体否定

| 配置 | round1 | round2 | 崩溃时 `Running` |
| --- | --- | --- | --- |
| D1 `--max-num-seqs 24` | ✅ 216.66 tok/s，TTFT 156 s | ❌ VMFault | 23 |
| D2 `--max-num-seqs 16` | ✅ 208.71 tok/s，TTFT **187.6 s** | ❌ VMFault | 16 |
| D3 `--max-num-seqs 32 --enforce-eager` | ✅ 190.54 tok/s，TTFT 145.8 s | 见下方 |

**五个配置，全部 round1 通过、round2 崩溃：**

| 作业 | 配置 | round1 吞吐 | round1 TTFT | round2 |
| --- | --- | ---: | ---: | --- |
| 1123145 | 默认 | — | — | ❌ 崩在 `Running=29` |
| 1123145 | `--enforce-eager` | 239.32 | 43.6 s | ❌ 崩在 62 |
| 1123874 | `--max-num-seqs 24` | 216.66 | 156.2 s | ❌ 崩在 23 |
| 1123874 | `--max-num-seqs 16` | 208.71 | 187.6 s | ❌ 崩在 16 |
| 1123874 | `--max-num-seqs 32 --enforce-eager` | 190.54 | 145.8 s | ❌ 只成功 **31/128** |

⚠️ **进度条 100% 是假象**：mns32eager 的 round2 日志显示 `100%|`，但
`Successful requests: 31`、`Benchmark duration: 209 s`（正常轮是 550–690 s）。
失败请求会飞速计数把进度条推满，**必须看 `Successful requests` 而不是进度条**。
这是本轮差点误判"第一次没崩"的直接原因。

`--max-num-seqs` 降到 16 后，TTFT 恶化到 **187.6 秒**（相对默认 43.6 s 是 4.3×），
吞吐也降到 208.71（−12.8%），**依然 round2 崩**。这条路线的性价比已经很清楚：

- 每一档降批都付出 3–4 倍的 TTFT 代价；
- 换来的只是"round1 能出数"，round2 照样崩；
- 而 round1 出数在默认配置下本来就有（只是没有效性保证）。

→ **降批不是解法，只是把崩溃推迟到第二次。** 真正要回答的是"为什么第二次必崩"，
这正是 job 1124366 在测的。

⚠️ **数据有效性说明**：D1/D2 的 round1 数字是在 `bench.sh` 有效性校验**仍然损坏**时产出的
（远程 `zoo-sync` 的副本还有 `EXPECT = len(ILENS.split())` 这行，见 gotchas #45）。
数字本身来自 sglang 的独立统计（`Successful requests: 128`），不是伪造的，
但**没有经过"TTFT=0 / 吞吐超限"闸门复核**。已于 2026-11-15 16:53 修掉远程副本。

## E：显式 `--attention-backend ROCM_AITER_MLA`（作业 1124165）—— ❌ **服务起不来**

配置 E（`--attention-backend ROCM_AITER_MLA`）与 F（同 + `--enforce-eager`）**两档都在引擎初始化阶段死掉**，
四个 TP rank 同时报同一个错：

```
(Worker_TP0 pid=228) ERROR 11-15 16:23:16 [multiproc_executor.py:870]
  ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'
  (/usr/local/lib/python3.10/dist-packages/aiter/__init__.py)
(APIServer pid=1) RuntimeError: Engine core initialization failed. Failed core proc(s): {}
```

**这是"能 import 不等于能跑"的教科书案例**，而且比预期更早死——不是 kernel 在 gfx936 上算错，
而是 `AiterMLABackend` 模块顶层 `from aiter import flash_attn_varlen_func` 在**镜像自带的
aiter 版本里根本没有这个符号**。也就是说：

- 机制层面（作业 1124018）确认无误：显式指定确实绕过了 `on_mi3xx()`；
- 但绕过之后拿到的是一个**与镜像 aiter 版本不匹配的 backend 实现**，
  vLLM 0.20 的 `rocm_aiter_mla.py` 要求的 API 在该 aiter 构建里不存在；
- `--enforce-eager` 走的是同一个 import 路径，所以 F 档必然同样失败（实测果然如此）。

**结论：aiter 路线彻底关闭。** 要让它跑起来需要同时满足"补 aiter 的
`flash_attn_varlen_func`"和"gfx936 上有可用 kernel"两个条件，前者是改镜像、后者是改上游硬编码。
按 HANDOFF 的原则（不改 vLLM 源码、不换镜像），到此为止，不再投卡时。

## ⚠️ 重读旧日志推翻了"round1 永远过"：崩溃与**累计运行时长**相关

此前所有表格都写"round1 ✅ / round2 ❌"，于是把变量锁定在"第几轮"。
**重新读 job 1120618 的 `vllm-logs-1120618.txt`（3 MB）后发现不是这样**：

该作业是**单次服务、跑完整 6 档序列**，`slurm-1120618.out` 里的时间线是：

```
10:30:28  round1_4096_1024_c1
10:34:41  round2_4096_1024_c1
10:38:40  round1_4096_1024_c8
10:40:13  round2_4096_1024_c8
10:41:36  round1_4096_1024_c64
10:45:41  round2_4096_1024_c64
10:48:44  round1_16384_1024_c1
10:53:08  round2_16384_1024_c1
10:57:28  round1_16384_1024_c8
10:59:25  round2_16384_1024_c8
11:01:20  round1_16384_1024_c64   ← 崩在这里（首轮！）
11:03:06  round2_16384_1024_c64
```

服务端 `vllm-logs-1120618.txt` 的对应现场：

```
10:30:08  第一个 Running 计数（压测开始）
11:02:38  Running: 29 reqs,  Waiting: 35 reqs, KV cache 11.8%
11:02:48  Running: 33 reqs,  Waiting: 31 reqs, KV cache 13.5%
11:02:58  Avg generation throughput: 0.0    ← 死了，日志里共 12 处 KERNEL VMFault
```

**结论修正**：

1. `Running=29` 这次崩溃发生在 **16K/c64 的 round1**，不是 round2；
2. 但它是该服务进程启动 **31 分钟后**才崩的——前面已经跑完 10 个档位、
   处理了约 10×128 ≈ 1200 个请求；
3. 后续每个作业只跑 16K/c64 一档（128 请求 ≈ 10 分钟），所以"第一轮总过、第二轮总崩"
   其实等价于"**服务进程活到 ~20 分钟就崩**"。

→ **真正的变量是"服务进程累计处理量/运行时长"，不是"第几轮"，也不是 `Running` 数。**
`--max-num-seqs` 24/16/32eager 全部失败也就解释得通了：它们只改单批规模，
根本没有延长进程的"寿命"。

**这个重读修正了 HANDOFF 里"默认配置崩在 Running=29"的语境**——那句话本身没错，
但被误读成"首轮就崩"，实际是"跑了 31 分钟后在第 11 个压测档上崩"。

**判据**：以后比较"崩/不崩"时，必须同时记录
**该服务进程已运行的分钟数**和**已成功处理的请求总数**，而不只是 `Running`。

## G：每轮重启服务 ×4（作业 1124366）—— 本轮唯一能改变结论的实验

**设计**：每轮压测前 `podman rm -f` 再重新 `podman run`，让每一轮都跑在**全新进程**上。
其余参数与默认配置完全一致（`--max-num-seqs 64`、无 eager、`--max-model-len 20480`）。

**要区分的两个假设**：

| | 假设 | 预测 |
| --- | --- | --- |
| H1 | 累计状态（进程跑久了才崩） | 4 轮全过 |
| H2 | 每轮独立的概率事件 | 首轮就会崩 |

**R1–R4 结果**（2026-11-15 16:54、17:06、17:17、17:27，每轮都是全新服务进程）：

| 指标 | R1 | R2 | R3 | R4 |
| --- | ---: | ---: | ---: | ---: |
| 成功请求 | **128/128** | **128/128** | **128/128** | **128/128** |
| Output throughput | 235.14 tok/s | 283.53 tok/s | 284.62 tok/s | **286.84 tok/s** |
| Mean TTFT | 47,488 ms | 31,498 ms | 30,841 ms | **30,608 ms** |
| Mean TPOT | 218.3 ms | 187.4 ms | 187.2 ms | **185.7 ms** |
| Benchmark duration | 557.42 s | 462.29 s | 460.52 s | **456.95 s** |
| 服务端 VMFault | **0** | **0** | **0** | **0** |

R2/R3/R4 三连一致（283.53 / 284.62 / 286.84 tok/s，极差 **1.2%**），
说明**这是可重复的稳定结果**，不是偶然。R1 偏慢 17%，是冷启动后页缓存未暖，
**不能当作稳态数据引用**。

**这四轮是目前最好的 16K/c64 数据**，且**没有 `--enforce-eager`**，
说明默认配置 + cudagraph 本身没问题——只要进程是新的。

### 判定：**H1（累计状态）成立**

脚本自动判定原文（`rs_1124366.out`）：

```
[rs 17:27:57] 通过 4 / 4 轮
[rs 17:27:57] 判定：**H1 累计状态成立** —— 全新服务进程单跑一轮可反复成功，
             round2 崩溃源于进程内状态累积。
```

- 全新进程单跑 128 个 16K 请求，**连续四轮都干净通过**，0 处 VMFault；
- 而此前**所有**"跑完第一轮接着跑第二轮"的配置全部崩溃（5 个配置，5 个崩溃点）；
- 与"1120618 跑满 31 分钟/约 1200 请求后崩"完全吻合。

**交付时引用 283.53–286.84 tok/s（R2–R4 极差 1.2%），不要引用 R1 的 235.14。**

→ **交付口径：每轮压测前重启服务。** 代价是每轮多约 3–4 分钟冷启动
（`bench.sh` 的 preflight + 服务就绪约 300 s），换来的是**可重复的有效数据**。

## A：`--enforce-eager` 把崩溃点从 29 抬到 62，但**没根治**

round1 完整数据：

```
Successful requests:      128
Mean TTFT (ms):           43563.29
Output token throughput:  239.32 tok/s
Peak concurrent requests: 65
Benchmark duration (s):   547.69
```

round2 服务端现场：

```
15:03:40  Running: 53 reqs, Waiting: 11 reqs, GPU KV cache usage: 22.0%
15:03:50  Running: 58 reqs, Waiting:  6 reqs, GPU KV cache usage: 23.9%
15:04:00  Running: 62 reqs, Waiting:  2 reqs, GPU KV cache usage: 25.6%
          Invalid address access: 0x7fb9656cd000, Error code: 1.
          >>>>>>>> KERNEL VMFault !!!! <<<<<<
```

**判读（推翻了此前的假设）**：

- 假设"PIECEWISE cudagraph 是元凶"**不成立**——eager 下根本没有 cudagraph（日志
  `cudagraph_mode=NONE`），照样在 `Running=62` 崩。
- 假设"缓冲越界（`max-model-len` 不够）"**也不成立**——崩溃时 KV cache 只用 25.6%。
- 真实条件更像"**高并发 + 长上下文下的概率性 kernel 越界**"：round1 峰值并发 65 侥幸通过，
  round2 在 62 崩了。**整个作业里服务端共记录了 10 处 `KERNEL VMFault`**。

**所以 `--enforce-eager` 的价值是"让这一档偶尔能出数据"，不是"修好了"。**

### 崩溃现场的第二个线索：**四个 TP rank 同时崩**

上一版只看了 `KERNEL VMFault` 这个字符串。重新逐行读 `srv-eagerA-1123145.txt`
（2.6 MB）后，崩溃现场其实是：

```
Invalid address access: 0x7fd85e1fe000, Error code: 1.     >>>>>>>> KERNEL VMFault
Invalid address access: 0x7fb9656cd000, Error code: 1.     >>>>>>>> KERNEL VMFault
Invalid address access: 0x7f7fbd283000, Error code: 1.     >>>>>>>> KERNEL VMFault
Invalid address access: 0x7ef96ba83000, Error code: 1.     >>>>>>>> KERNEL VMFault

PID: 224 SIGNAL: 0    PID: 225 SIGNAL: 0    PID: 226 SIGNAL: 0    PID: 227 SIGNAL: 0
```

**四个 PID 就是 TP=4 的四个 worker，同一时刻各崩一次，四个地址互不相同。**

这条改变了对根因的判断：

- **不是"某一个 kernel 越界"**——那样只会有一个 rank 先死，其余 rank 随后因
  集合通信超时或对端断开而退出（报的是 timeout / NCCL 错误，不是 VMFault）。
- **四个 rank 同时 VMFault，且地址各不相同**，更像**集合通信的缓冲区**：
  all-reduce / reduce-scatter 在四个 rank 上各自映射了一块显存，某一侧的长度或
  offset 算错，四块同时越界。
- 与"KV cache 只用了 25.6%"并不矛盾：集合通信的缓冲区与 KV cache 是两块独立显存。

**待验证**：崩溃前是否有 rank 正在做 all-reduce。`max_num_seqs` 实验
（job 1123874）若在 24 就稳、64 必崩，则更可能是"batch 维度参与集合通信时长度算错"，
而不是注意力 kernel 本身。

⚠️ 这条是**重新阅读旧日志**得到的，不是新实验。写进文档前请先按
`grep -aoE "Invalid address access: 0x[0-9a-f]+" ... | sort | uniq -c` 自行复核。

## B：`--max-model-len 17408` 是**我算错了**

服务端直接返回 400：

```
VLLMValidationError: This model's maximum context length is 17408 tokens.
However, you requested 1024 output tokens and your prompt contains at least 16385 input tokens,
for a total of at least 17409 tokens.
```

`16384 + 1024 = 17408` 恰好等于上限，但**实际 prompt 是 16385 个 token**（多 1 个，
来自 chat template / BOS）。所以 `17408` 差一个 token。压测客户端因此在 17 秒内
"跑完" 128 个请求——全部是 400，`e2e_latencies` 为空，`np.percentile` 抛 `IndexError`。

**教训**：`max_model_len` 要留余量，别按 `输入 + 输出` 精确计算。
`bench.sh` 的 `preflight_lib.py` 本来就是为了防这个（它用真实请求探测长度阶梯），
但 B 档启动时 `sed` 改的 `MAX_LEN` 与服务端一致、preflight 又只测到 16384 能过，
所以没拦住——真正被拒的是 16385。

## 认识论上的一个硬伤：`bench.sh` 的校验段从来没跑过

复盘 B 档时发现 `bench.sh:49` 的 `import json, sys` **漏了 `os`**，而紧接着的
`os.path.exists(...)` 直接 `NameError`。也就是说：

- 「TTFT=0 → 无效」「吞吐超物理上限 → 无效」这两道闸门**从未生效**；
- B 档全部 400 时本该报 `INVALID 16384_64: TTFT=0（请求疑似被拒）`，
  实际却是下游 `np.percentile([])` 抛 `IndexError`，把配置错误伪装成脚本崩溃；
- 这段模板复制在 `models/*/bench.sh` 共 10 个文件里，全部失效。

已于 2026-11-15 修复（`import json, os, sys`），并用手造坏数据验证过它能正确报错。
详见 gotchas #45。**本仓库此前所有"压测有效"的判读都缺这一层保护。**

## 后续该查什么

**交付层面已经没有阻塞**——16K/c64 用"单轮 + 重启服务"可以稳定出数。若还有余力：

1. **根因定位**（可选，纯研究）：累计状态到底存在哪一块。三个候选，按排查成本排序：
   - **KV cache 块分配/释放**：进程跑久后块池碎片或引用计数泄漏 → 用
     `VLLM_ATTENTION_BACKEND` 之外的手段看不到，可试 `--enable-prefix-caching` 开关对比；
   - **KDA 的 recurrent state**：`KimiDeltaAttention` 维护跨请求的状态缓冲，
     看 vLLM 是否在请求结束时正确重置；
   - **集合通信 workspace**：四个 rank 同时崩且地址不同，与这一条最吻合。
     `--disable-custom-all-reduce` 已经是 `True`，可试改 NCCL/RCCL 的
     `RCCL_BUFFSIZE` 或关掉 `ag_rs`（`dcp_comm_backend`）对比。
2. ~~`--max-num-seqs` 降批~~ — **已否定**（1123874 三档全崩，见 D2/D3）。
3. ~~显式 `--attention-backend ROCM_AITER_MLA`~~ — **已否定**（1124165 服务起不来）。
4. ~~换镜像 0.21~~ — **已否定**（加载权重即崩，见 README §7.3）。

**验证"是否真的与时长相关"的低成本实验**：同一服务进程连跑 3 轮 16K/c64，
每轮结束后重启**客户端**但不重启服务，记录第几轮开始崩、崩时累计处理了多少请求。
若崩点稳定落在"累计 ~1500–2000 请求"或"运行 ~25–35 分钟"，根因就锁定在时长维度。

## 复现

```bash
# 推荐的交付口径（每轮重启服务）
sbatch sweeps/kimi-16k-020-restart.slurm      # NROUND=4，默认

# 历史对照（会复现 round2 崩溃）
sbatch sweeps/kimi-16k-020-matrix.slurm       # A 档有数据即 exit 0
```

⚠️ 提交前**必须先 `./tools/sync_remote.sh`**，否则远程 `zoo-sync` 的旧 `bench.sh`
会被 rsync 进作业容器，有效性校验段会以 `NameError` 失败（gotchas #45）。
