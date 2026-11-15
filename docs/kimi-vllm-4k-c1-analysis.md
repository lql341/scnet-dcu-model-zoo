# Kimi × vLLM 4K/c1 落后 61%：机理分析（2026-11-15）

> **一句话结论**：这**不是** kernel 效率问题，是**每步 ~24 ms 的固定开销**——
> vLLM 的 TPOT 对 batch 和 context 都近乎不敏感，而 SGLang 的随请求数增长。
> 机理指向「TritonMLA 无法整图捕获 + TP=4 逐层集合通信」，属引擎/内核特性。
> **2026-11-15 实测补充**：`--async-scheduling` 让 4K/c8 吞吐 +23.2%、TTFT −61.7%，
> 但单流 TPOT 仅 −3.2%，24 ms 主体仍在。

## 1. 现象

同模型（Kimi-Linear-48B-A3B-Instruct）、同卡（BW1000 ×4 TP=4）、同精度（bf16）、
同压测客户端（`sglang.benchmark.serving`）：

| 档位 | vLLM 0.20 | SGLang 0.5.10 | vLLM/SGLang |
| --- | ---: | ---: | ---: |
| 4K/c1 | 40.73 | 104.67 | **−61%** |
| 4K/c8 | 304.18 | 353.53 | −14% |
| 4K/c64 | **858.08** | 839.00 | **+2%** |
| 16K/c1 | 37.36 | 50.80 | −26% |
| 16K/c8 | **197.83** | 119.21 | **+66%** |
| 16K/c64 | **284.62** | 234.17 | **+22%** |

低并发落后、高并发反超——这个反差是本条目最值得解释的现象。

## 2. 关键测量：TPOT 的分解

从各自的 `bench/round2.jsonl` 取 `mean_tpot_ms`：

| 配置 | vLLM TPOT | SGLang TPOT |
| --- | ---: | ---: |
| 4K/c1 | **24.18 ms** | 9.31 ms |
| 4K/c8 | 24.96 ms | 20.97 ms |
| 16K/c1 | 25.23 ms | — |

**两条曲线的形状完全不同**：

- **vLLM**：4K/c1 24.18 → 4K/c8 24.96 → 16K/c1 25.23。
  对 batch 和 context 都**几乎不动** → 拟合为 `24.07 ms + 0.111 ms/req`
- **SGLang**：4K/c1 9.31 → 4K/c8 20.97（+125%）。
  拟合为 `7.64 ms + 1.67 ms/req`

**含义**：vLLM 每步先付 ~24 ms，之后加请求几乎免费；
SGLang 每步只付 7.6 ms，但每个请求要加 1.67 ms。
所以并发越高，SGLang 的边际成本越吃亏——这就是 vLLM 在 c64 反超的原因。

## 3. 24 ms 从哪来

24 ms ÷ 27 层 ≈ **0.89 ms/层**。这个量级对得上两件事：

1. **逐层 eager kernel launch**（每层几十个 kernel，每个几十微秒的发射开销）
2. **TP=4 的逐层 all-reduce 同步**

而**不是**：
- 显存带宽（4K→16K 的 KV 只让 TPOT 涨 1 ms，说明注意力本身不是瓶颈）
- prefill（TTFT 只解释 4K/c1 缺口的 **~1%**：vLLM 3.25 s vs SGLang 2.06 s，而总时长差 122.9 s）

### 机理线索

该条目 README §7 记录了一条启动日志：

```
[compilation.py:1370] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with
  TritonMLABackend backend (support: AttentionCGSupport.NEVER); setting cudagraph_mode=PIECEWISE
```

`AttentionCGSupport.NEVER` 意味着 **MLA 层的 decode 注意力永远进不了整图**，
只能 piecewise 捕获，其余部分每步 eager 发射。Kimi 是 MLA 架构（7 层 KimiMLAAttention），
所以这条限制直接命中。

## 4. 这个缺口能不能补

**单流天花板**：如果 24 ms 完全是固定开销，vLLM 的 4K/c1 上限 ≈ 1000/24.2 ≈ **41 tok/s**。
实测 40.73，已经贴着这个天花板——**说明瓶颈确实是那 24 ms，而不是某个能调好的参数**。

### 2026-11-15 实测结果（作业 1128368）

**先决信息把第一项假设直接作废**：该条目的 `compilation_config.mode` 本来就是
**`VLLM_COMPILE`**，不是 `NONE`。所以「开启编译配置」这条路不存在——已经是那个模式了。

| 变体 | 4K/c1 吞吐 | 4K/c1 TPOT | 4K/c8 吞吐 | 4K/c8 TTFT |
| --- | ---: | ---: | ---: | ---: |
| E0 基线 | 39.78 | 24.35 ms | 239.16 | 3949 ms |
| **E2 `--async-scheduling`** | **41.63（+4.6%）** | **23.58 ms（−3.2%）** | **294.69（+23.2%）** | **1511 ms（−61.7%）** |

**结论**：

1. **`--async-scheduling` 在并发档有效且幅度很大**：4K/c8 吞吐 +23.2%，
   同时 TTFT 降低 61.7%。这是本轮 vLLM 侧唯一确认有效的参数。
2. **单流档改善有限**：4K/c1 吞吐 +4.6%、TPOT 仅 −3.2%（24.35 → 23.58 ms）。
   24 ms 固定开销的主体**未被消除**，与"逐层 eager 发射 + TP 集合通信"的推断一致。
3. 因此单流天花板问题（≈41 tok/s）依然成立：本次实测 41.63 tok/s 正好贴着该上界。

### 已排除的方向

| 参数 | 结论 |
| --- | --- |
| `--compilation-config` | ❌ **作废**：该条目已是 `VLLM_COMPILE` 模式 |
| RCCL 调优 | 未测；`disable_custom_all_reduce` 已 True，走 RCCL |

实验脚本：`sweeps/kimi-vllm-4k-opt.slurm`（作业 1128188 首次失败、1128368 完成）。

### 已确认无效（不要重复）

`--enforce-eager`（有害）、`--kv-cache-dtype fp8`（无 fp8 GEMM kernel）、
`--attention-backend ROCM_AITER_MLA`（镜像缺 `flash_attn_varlen_func`）、
`VLLM_USE_V1=0`、`VLLM_ROCM_USE_AITER=1`、`VLLM_ATTENTION_BACKEND=FLASH_ATTN`、
0.21 镜像（加载权重即崩）。

**两个"看起来该调"但实际是 no-op**：`--enable-chunked-prefill`（vLLM 默认已开）、
`--disable-custom-all-reduce`（该镜像已默认 True）。

## 5. 一个必须注意的工程陷阱

该条目的 `container/submit.slurm` **不引用 `$EXTRA_ARGS`**（grep 计数 0，参数内联在 75-79 行），
`run.sh` 里还留着 SGLang 启动器残留。

**改 `env.sh` 完全无效。** 任何参数扫描必须照 `sweeps/kimi-16k-020-maxseq.slurm` 的
`run_config "$extra"` 模式写，否则会像 gotcha #55 那样静默测到默认配置。

## 6. 与 gotcha #21 的区别（重要）

`docs/gotchas.md` #21 讲的是 **MiMo-9B × vLLM 的 16K 档慢 5-7 倍**，
根因是 DCU 缺 paged-attention kernel，**TPOT 从 4K 的 47.9 ms 涨到 16K 的 119.3 ms**——context 相关。

Kimi 这个是 **TPOT 恒定 24 ms**——context 无关。

**两者根因不同，不要套用同一个结论。**

## 7. 复核方式

```bash
# TPOT 原始值
python3 -c "
import json
for eng in ('vllm','sglang'):
    p=f'models/moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__{eng}/bench/round2.jsonl'
    for l in open(p):
        d=json.loads(l)
        print(eng, d['random_input_len'], 'c%d'%d['max_concurrency'],
              'tpot=%.2f'%d['mean_tpot_ms'], 'out=%.2f'%d['output_throughput'])
"
```
