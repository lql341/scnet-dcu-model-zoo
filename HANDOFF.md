# HANDOFF —— 下一段推进的交接文档

> 版本：2026-11-15（2026-11-21 复核修订）　**本文件是唯一有效的交接文档**（旧的 2026-11-14 版已删除，历史细节见 `docs/ARCHIVE-SUMMARY.md` 与 `docs/REPORT.md`）
> 项目性质：**能力沉淀，不是比赛提交**（2026-11-15 决策）。所有产出的第一价值是"下一个人能复用"。
> 归档现状见 §0「归档」行与 §2.6：原 `luql:/public/home/luql/archive/zoo-dcu-2026-11/` 目录在 2026-11-21 OpenAPI 复核时已不可见，镜像本体仍散落在原账号（详见 §2.6）。

---

## 0. 三分钟上手

| 项 | 现状 |
| --- | --- |
| 条目 | **20 个**（14 个不同模型 × 卡型 × 框架 × 精度） |
| 官方评测 | **9 个条目全绿，935/935 条真实产物**（官方 9 份清单 9/9 覆盖） |
| 跨引擎基准 | **8 个条目**（MiMo / Kimi / Recon2Reason / RoboBrain，各 SGLang + vLLM × 6 档） |
| 自建运行时 | **1 个**（Trident-Qwen3VL-2B：OpenAI 兼容 embedding 服务，两引擎均不支持该架构） |
| 阻塞 | **4 个**（Trident × SGLang/vLLM 架构未注册；Recon2Reason/RoboBrain × SGLang 于 2026-11-15 封存，见 gotcha #58） |
| 踩坑 | **59 条**（`docs/gotchas.md`，按隐蔽度排序） |
| 归档 | 历史口径 5 镜像 55 GB（规范前 31 GB）+ assets + sources + products；**2026-11-21 实测原 `archive/` 目录已不可见，镜像本体仍在原账号——见 §2.6**；`SHA256SUMS` 11/11 为历史校验结果 |
| 代码同步 | GitHub `lql341/scnet-dcu-model-zoo` = 本地（每次改完 `tools/publish.sh`）。GitLab/sourcefind 远端已于 2026-11-21 按用户决定移除，不再维护 |

**本段最重要的四个教训**（写在这里，防止新会话重犯）：

1. **先看调度器的 `AllocTRES`，再看容器，最后看 torch** —— `--gres=dcu:N` 可能被静默降级成 1 张卡，
   每张 DCU 需 ≥8 CPU（`-c 64` 配 8 卡）。见 gotchas #30/#31。
2. **`2>/dev/null` 会把"拷仓库失败"变成"缺 Python 包"** —— `WORK` 与 `REPO` 相同会让 `cp` 把自己拷进自己，
   于是 `patches/` 没进容器、报出完全无关的 `ModuleNotFoundError: dashscope`。见 gotchas #33。
3. **报告失败前先确认"你测的是系统行为还是自己的超时"** —— 压测客户端默认只等 60 s，
   而 48B 模型冷启动 410 s；不改超时永远看不到真因（GPU VMFault）。见 gotchas #36。
4. **别把"第二轮崩"当成"批大小问题"** —— 2026-11-15 为此连开 3 个作业降 `--max-num-seqs`，
   全部失败。真因是服务进程的**累计状态**：重启一次就能稳定出数（283.53 tok/s）。
   教训是**先做一个能分离变量的实验**（每轮重启），而不是沿着最直觉的解释连打三枪。
   见 gotchas #51。

---

## 1. 本轮决策记录（2026-11-15，用户拍板）

| # | 决策 | 影响 / 落地方式 |
| --- | --- | --- |
| 1 | **投入 kernel 主线** | 见 §3（P0），这是唯一能同时解锁"vLLM 长上下文"和"量化"的路径 |
| 2 | 量化暂不做 | 见 `docs/quantization-branch.md`：AWQ 慢 5×、fp8 无 kernel，先补 kernel 才有意义 |
| 3 | T01/T02 官方材料不管了 | 精度自评保持 `[BLOCKED]`，不填估算值；Recon2Reason/RoboBrain 维持"链路 + 基准"口径 |
| 3b | **官方澄清（2026-11-15）**：T01–T02 验收以官方公开基线为准，不再单独下发配置文件 | **已查证，结论是解不了**：① 官方公开分数是 8-shot/5-shot、base/EM 口径，`verify.py` 是 0-shot greedy，**相减无意义**；② MiMo 官方指标里**没有** MMLU/GSM8K，Kimi 的 GSM8K 86.3 是 **base 8-shot**；③ GSM8K/MMLU 数据集本集群**取不到**（HF/GitHub 超时，modelscope 镜像无数据文件）。→ §6 保持 `[BLOCKED]`，理由改为"口径不可比+数据不可得"。详见 `docs/t01-t02-baseline-investigation.md` |
| 4 | Harbor 不手动操作，镜像就放 `luql` 家目录 | 已有 tar.gz + `SHA256SUMS` 兜底；风险（平台清账号）记在 §6 |
| 5 | **目的不是比赛**，重点是文档与交付物 | 不再做"提交格式"对齐；把每条经验写进 `docs/gotchas.md` 与条目 README |
| 6 | **能做的都做**：Kimi / Qwen-VL / Trident 等阻塞逐个解除 | 见 §4（P1）与 §5（P2） |
| 7 | 本会话收尾 = 写这份交接 | 新会话从这里开始 |

---

## 2. 已交付清单（**不要重做**，只需引用）

| 类别 | 数量 | 位置 |
| --- | --- | --- |
| 官方条件产物 | 935 条 | `archive/products/<model>/outputs/`，元数据同步在 `models/*/official-eval/` |
| 重跑补档产物 | AuK 120 wav + Qwen-Image 180 PNG（2026-11-15） | `products/auk/`、`products/qi/` |
| 6 档基准 | 8 条目 × 6 档 × 2 轮 | `models/*/bench/{summary.md,round1.jsonl,round2.jsonl}` |
| embedding 分档 | 6 档（items/s 口径） | `models/OpenBMB__Trident-Qwen3VL-2B__bw1000__transformers/bench/` |
| 镜像 | 5 个 tar.gz（55 GB） | `archive/images/`，命名 `<仓库名>_<tag>.tar.gz` |
| 源码 | AuK / Wan2.1 / OmniWeaving | `archive/sources/` |
| 轮子 | cp310 多套（auk-fix / qi2 / omni-fix / wan-fix / trident） | `archive/assets/` 与 `~/wheels/` |
| 经验 | 59 条踩坑 + 4 条有效硬约束（1 条已推翻） | `docs/gotchas.md`、`docs/REPORT.md` §4 |

**关键性能数字（引用以此为准）**

| 场景 | 数字 |
| --- | --- |
| MiMo-9B 4K/c64 | SGLang **998.05** vs vLLM 489.52 tok/s |
| Kimi-48B 4K/c64 / 16K/c8 | 839.00 vs **858.08** / 119.21 vs **197.83** |
| Recon2Reason-4B 4K/c64 | 719.64 vs **845.30** |
| RoboBrain2.5-8B 4K/c64 | 543.80 vs **667.69** |
| OmniWeaving 单条 | 单卡 1475 s → **4 卡 SP 662 s（2.23×）** |
| Wan2.1 单条 | 4 卡 FSDP 1116 s → **8 卡 FSDP 1003 s（1.11×，已饱和）** |
| Trident embedding | 512 tok **22.8 条/s**、2K 15.2、4K 8.9（并发 1→8 仅 +2%） |

---

## 2.5 2026-11-15 下午-晚间：本轮新增结论（**已完成，可引用**）

### 2.5.1 吞吐优化的实收

| 条目 | 改动 | 实测收益 | 数据来源 |
| --- | --- | --- | --- |
| **Kimi × SGLang** | `--chunked-prefill-size 4096` 进 `EXTRA_ARGS` | 16K/c8 **+10.5%**（124.31→137.30）、16K/c64 +2.7%（222.45→228.40）、c64 TTFT −10% | 作业 1126055，同节点 A/B |
| **Qwen3-ASR × vLLM** | `MAX_NUM_SEQS` 16→128、`WORKERS` 4→32 | 60 条 **30s→26s（−13%）**，CER/WER 指标不变 | 归档产物复核 |

> ⚠️ Kimi 的 c64 +2.7% 与本档历史波动（±5%）重叠，**不能算确定收益**；
> `meta.throughput.16k_c64` 因此保持 234.17 未改，只把 chunked 的影响写进 README §7.4。

> ⚠️ ASR 的"精度零变化"措辞已修正：逐条比对归档产物后，60 条中有 **1 条**
> （`1089-134686-0024.txt`）在 64 并发档多出一个逗号。CER 上看不出，但**不是逐字节一致**。

### 2.5.2 Kimi × vLLM 的 4K/c1 缺口已查清机理（未优化）

4K/c1 是 **40.73 tok/s，比 SGLang 的 104.67 低 61%**。本轮查明原因**不是** kernel 效率：

| 量 | vLLM | 拟合 |
| --- | ---: | --- |
| TPOT 4K/c1 | 24.18 ms | vLLM = **24.07 ms + 0.111 ms/req** |
| TPOT 4K/c8 | 24.96 ms | 对 batch 几乎不敏感 |
| TPOT 16K/c1 | 25.23 ms | 对 context 几乎不敏感 |
| SGLang 4K/c1 | 9.31 ms | SGLang = 7.64 + 1.67 ms/req |

即 vLLM 每步先付 **~24 ms 固定开销** ≈ 27 层 × 0.89 ms/层，指向**逐层 eager kernel launch + TP all-reduce**。
机理线索：`TritonMLABackend` 的 `AttentionCGSupport.NEVER` 迫使 `cudagraph_mode=PIECEWISE`，
7 层 MLA 的 decode 注意力永远进不了整图。

**推论**：vLLM 单流天花板 ≈ 1/0.0242 ≈ **41 tok/s**，任何注意力/上下文参数都突破不了。
→ **这不是 gotcha #21 那个"缺 paged-attention 内核"**（那个是 context 相关，MiMo 的 TPOT 47.9→119.4）。

**实测结果（作业 1128368）**：

| 变体 | 4K/c1 吞吐 | 4K/c1 TPOT | 4K/c8 吞吐 | 4K/c8 TTFT |
| --- | ---: | ---: | ---: | ---: |
| 基线 | 39.78 | 24.35 ms | 239.16 | 3949 ms |
| **`--async-scheduling`** | 41.63 (+4.6%) | 23.58 ms (−3.2%) | **294.69 (+23.2%)** | **1511 ms (−61.7%)** |

- **并发档大幅改善**：4K/c8 +23.2%，TTFT 降 61.7%；
- **单流档改善有限**：TPOT 仅 −3.2%，24 ms 固定开销主体仍在，天花板 ≈41 tok/s 依然成立；
- `--compilation-config` 这条**作废**——实测 `compilation_config.mode` 本来就是 `VLLM_COMPILE`。

### 2.5.3 Recon2Reason / RoboBrain × SGLang 已封存

见 gotcha #58。服务可启动但 warmup 时 `_fwd_grouped_kernel_stage1` 触发 KERNEL VMFault。
已排除节点/参数/显存/cuda-graph/架构/镜像整体损坏（**Kimi 用同一镜像当日跑通**）。
最可能根因：`tvm-ffi` JIT KV-cache 路径的环境回归。
**两个条目的 vLLM 兄弟条目不受影响**（2026-11-12 已产出完整 6 档数据）。

### 2.5.4 文档一致性：10 项 🔴 级问题已修

本轮审计发现并修复的典型问题（都记在 gotcha #54/#55）：

- 3 个 `__vllm` 条目的 `run.sh` 是 **SGLang 启动器残留**，引用未定义的 `$MAX_RUNNING_REQUESTS`
- `MiMo × vLLM` 的 `submit.slurm` 保留已废弃的 `--enforce-eager`，与文档声明的"graph 开启"矛盾
- `Kimi × SGLang` README 四处仍写 `--disable-cuda-graph`，而 §7.4 已证明"关掉 graph 掉 31%"
- `INDEX.md`「Kimi × vLLM ❌ 框架不支持」已被该条目 6 档数据推翻（错误源头在 `tools/gen_index.py` 硬编码）
- `Wan` 的 `FSDP` 默认值 0（分片）→ 1（§7.1 实测分片单卡 OOM）

### 2.6 归档实测复核（2026-11-21，scnet-hpc OpenAPI / region 20091 郑州）

**结论：原归档目录 `luql:/public/home/luql/archive/zoo-dcu-2026-11/` 已不可见，但镜像本体仍在原账号，未全部丢失。**
复核方法：SSH 后端无 zzeshell 密钥（本机只有 kseshell），改用 OpenAPI 文件服务（`scnet.py files`）遍历 12 个 region +
家目录全量索引（`INDEX.md`，2026-11-21 18:14 生成，9342 文件）。`archive/` 路径在所有 region 均返回 `File does not exist`，
家目录索引中亦无 `zoo-dcu-2026-11` 或镜像 tar 痕迹。

当前云端可见的归档本体（实测字节数）：

| 内容 | 位置 | 实测 |
| --- | --- | --- |
| SGLang 镜像 | `scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz` | 8083 MiB（≈7.9 GiB，sha256 f16125ba） |
| vLLM 0.20 镜像 | `scnethr2601a10/zoo-images/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz` | 8884 MiB（≈8.7 GiB，sha256 986f95a3） |
| Music3 镜像 | `scnethr2601a10/zoo-images/minimax-music3-bw-sglang_20261031-rc1.tar.gz` | 13093 MiB（≈12.8 GiB，sha256 b61b05bd） |
| vLLM 0.21 镜像 | `luql/work/imgstore/vllm021.tar`（**gunzip 后未压缩态**，`gunzip.done`） | 31265 MiB（≈30.5 GiB，压缩前口径 9.5 GB / sha256 c3b22bcc） |
| 依赖包集 | `scnethr2601a10/zoo-images/assets/`（5 个 tar.gz） | 165 MiB（wheels×3 + AuK-src 21.5M + harness 0.2M） |
| 生产元数据 | `luql/prodmeta/`（7 模型 json/csv） | ≈0.1 MiB |

历史口径：31 GB（2026-11-09/11 初版 3 镜像）→ 53/54/55 GB（2026-11-14 规范化后 5 镜像 + 产物 + 轮子 + 源码，
`docs/ARCHIVE-SUMMARY.md`/`docs/REPORT.md` 三个口径并存）。当前因 archive 目录移除，无法实测出与历史值相等的单一数字；
上表 3 个 tar.gz 合计 ≈29.4 GiB，加上未压缩的 vllm021.tar（≈30.5 GiB）与 assets，是现存本体的实测口径。
**引用大小时以此为准，并把"archive 目录已不可见、本体在原账号"一并说明，不要只写 55 GB。**

---


### 3.1 先做 0 卡时闸门（**没通过就不要投人力**）

在一个 1 卡作业里跑（镜像：`vllm0.20` 与 `vllm0.21` 各一遍，命令见 §7），只回答四个问题：

```bash
# ① 引擎支持哪些 attention backend（决定能不能绕开出错的 kernel）
python3 -c "import vllm.platforms.rocm as r, inspect; print(inspect.getsource(r))" 2>/dev/null | head -60
grep -rn "attention_backend\|AttentionBackendEnum" $SITE/vllm/v1/attention/backends/*.py | head -20

# ② aiter 是否存在（ROCm 官方 kernel 库，MLA/fp8/attention 都在里面）
python3 -c "import aiter; print(aiter.__version__)" || echo "无 aiter"

# ③ hipBLASLt 的 fp8 支持（fp8 GEMM 的唯一现实来源）
ls -l /opt/dtk/lib/libhipblaslt* 2>/dev/null; /opt/dtk/bin/hipblaslt-bench --help 2>&1 | grep -i "fp8\|e4m3" | head -5

# ④ Kimi-Linear 走的是哪套 kernel（关键：它是线性注意力，不是 paged-attn）
grep -rn "short_conv\|linear_attention\|gdn_attention\|causal_conv1d\|fused_recurrent" \
  $SITE/vllm/v1/attention/backends/*.py $SITE/vllm/model_executor/models/kimi_linear*.py 2>/dev/null | head -20
```

**闸门判据**：② 有 aiter **或** ① 里存在一个能让 16K/c64 跑通的 backend → 继续；
三项全否 → 写进 `docs/gotchas.md` 并停手，不投卡时。

> **闸门已完成（2026-11-15，0 卡时，作业 1121990）→ 通过。**
> 结果见 `sweeps/results-kernel-gate-2026-11-15.md`。两条修正必须先看：
> 1. **0.21 镜像对 Kimi 不可用**：`vllm_hcu/.../parameter.py:158` 对 3D 权重无条件 `.t()`，
>    Kimi 的 `conv1d` 权重加载即崩（作业 1122012，服务未起）。**别再试 0.21。**
> 2. Kimi 的 **MLA 层确实走 `--attention-backend`**（日志 `Using TRITON_MLA ... ['TRITON_MLA']`），
>    只有 KDA 层绕开。所以下面"换 backend"这条对本模型**部分有效**，前一版"完全无效"的说法已更正。

### 3.2 三个子目标（按证据强度排序，建议这个顺序）

**A. Kimi-Linear 长上下文崩溃（证据最硬，优先级最高）**

- 现象（2026-11-15 实测）：Kimi × vLLM，16K/c64 时 worker 被 GPU 访存错误打死：
  `Invalid address access: 0x7f7e01015000, Error code: 1` / `KERNEL VMFault` /
  `Worker proc VllmWorker-3 died unexpectedly`；崩溃时 `num_running_reqs=33`，**无 OOM**。
  同配置 16K/c1、c8 完全正常（34.78 / 184.51 tok/s）。
- **强烈线索（不要重复上一段"缺 paged-attn"的错误判断）**：从 vLLM 启动配置的 `splitting_ops` 看，
  Kimi-Linear 走的是 `short_conv` / `linear_attention` / `gdn_attention_core` 这类**线性注意力**路径，
  与 paged-attention 无关。
  ~~`fla_core`（flash-linear-attention）轮子存在（`~/wheels` 里有 0.6.0）~~
  **该说法已于 2026-11-15 核查证伪**：`~/softwares/wheels`、`~/wheels`、归档 `assets/` 下
  **没有任何 `fla`/`fla_core`/`causal_conv1d` 轮子**；`docs/environment.md:71` 也早已记录
  "外部 `fla` 包不需要"（KDA kernel 由 vLLM 自带 `linear_attn` 承载）。不要按"补 fla_core"规划。
- **崩溃现场取证（2026-11-15，job 1120618 的 3 MB 服务端日志）**：
  崩溃发生在 `Running` 从 8 逐级升到 **29** 的瞬间——正好跨过 PIECEWISE cudagraph 的 `24→32` 桶。
  同日志还有：
  ```
  [rocm.py:547] Using TRITON_MLA backend out of potential backends: ['TRITON_MLA'].
  [compilation.py:1370] CUDAGraphMode.FULL_AND_PIECEWISE is not supported with
      TritonMLABackend backend (support: AttentionCGSupport.NEVER); setting cudagraph_mode=PIECEWISE
  ```
  → **MLA 那一半层确实经过 backend 选择器**，只是候选集当前只有 `TRITON_MLA`。
  → ~~cudagraph + 大 batch 是头号嫌疑~~ **该假设已于 2026-11-15 被 job 1123145 证伪**（见下）。
- **2026-11-15 实测结果（job 1123145，A/B/C 三配置）**：

  | 配置 | round1 | round2 | 判读 |
  | --- | --- | --- | --- |
  | A `--enforce-eager` | ✅ 128/128，239.32 tok/s，峰值并发 65 | ❌ **VMFault**（`Running=62`） | cudagraph 不是根因，只放大问题 |
  | B `--max-model-len 17408` | ❌ 全部 400 | — | **配置算错**（16384+1024 差 1 token），非崩溃 |
  | C `17408+eager` | 未跑 | — | 同 B |

  - **A 的结论**：eager 下没有 cudagraph（`cudagraph_mode=NONE`），照样崩 →
    "PIECEWISE cudagraph 是元凶"**不成立**。崩溃时 KV cache 仅 25.6%，也不是显存。
    更像"高并发 + 长上下文下的概率性 kernel 越界"：round1 峰值 65 侥幸过，round2 在 62 崩。
  - **P0-A 判据（两轮都出数）仍未达成。**

- **2026-11-15 下午追加（job 1123874 `--max-num-seqs` 降批）—— ✅ 已完成，全败**：

  | 配置 | round1 | round2 | 崩溃时 `Running` |
  | --- | --- | --- | --- |
  | D1 `--max-num-seqs 24` | ✅ 216.66 tok/s，TTFT 156 s | ❌ VMFault | 23 |
  | D2 `--max-num-seqs 16` | ✅ 208.71 tok/s，TTFT **187.6 s** | ❌ VMFault | 16 |
  | D3 `--max-num-seqs 32 --enforce-eager` | ✅ 190.54 tok/s，TTFT 145.8 s | ❌ 只成功 **31/128** | — |

  - 降批的代价全在 TTFT（43.6 s → 156 s → 187.6 s），吞吐只降 9–13%，
    **但一档都没换来 round2 不崩**。
  - **五个配置、五个崩溃点：29 / 62 / 23 / 16 / 31 个请求**——与 `Running` 数无关。
    两个假设被否定：~~"单批规模是根因"~~、~~"cudagraph 桶边界"~~。
  - ⚠️ D3 的 round2 进度条显示 `100%|`，但 `Successful requests: 31`——崩溃后失败请求
    飞速刷满进度条（gotchas #50）。**判断"跑完了"必须看 `Successful requests`。**
  - **重读 job 1120618 的日志后修正了一个误读**：此前记的"默认配置崩在 `Running=29`"
    其实发生在**单次服务跑完 10 个档位之后**（10:30 启动 → 11:02:58 崩，**运行 31 分钟**、
    约 1200 个请求），而且崩在 16K/c64 的 **round1**。
    → 所以"round1 永远过、round2 永远崩"只是表象，真正的变量是
    **服务进程的累计运行时长/处理量**（job 1124366 的 R1 进程活了 27 分钟、只处理 128 个请求，
    照样通过；所以更可能是"请求总数"而不是时钟时长——1120618 崩时约 1200 个请求）。
  - 剩下的主流解释：**进程内状态随累计处理量累积**——KV cache 块释放、KDA 的 recurrent state、
    集合通信 workspace 复用后的残留。`--max-num-seqs` 改不动它，因为它只影响单批规模。
- **P0-A 的新判据实验（job 1124366 `kimi-restart`，2026-11-15 16:17 提交）**：
  **每轮都重启服务**，各跑一整轮 16K/c64，共 4 轮。
  - 3~4 轮全过 → **累计状态假设成立**，交付口径为"每轮之间重启服务"；
  - 首轮就崩 → **概率事件假设成立**，与轮次无关，需换 backend 或降上下文。
  - 脚本：`sweeps/kimi-16k-020-restart.slurm`
- 起手动作（2026-11-15 实测后再次修正）：
  1. ~~`--enforce-eager`~~ 已跑：只把阈值从 29 抬到 62，**不是解法**；
  2. ~~`--max-model-len 17408`~~ 已跑：**这个值本身差 1 token**，要用 ≥17920；
  3. ~~把 MLA 切到 `FLASHMLA`~~ **已排除**：`vllm._flashmla_C` 在镜像里没编译，
     `FLASHINFER_MLA` 缺 `flashinfer` 包。可换的只有 `ROCM_AITER_MLA` /
     `ROCM_AITER_TRITON_MLA`（`validate_configuration` 实测 OK），但它们要求 aiter 可用。
  4. ~~aiter 的修法~~ **两道门都已查清，且第二道在 BW1000 上无解**：
     - 门 1：`/opt/dtk/bin/rocminfo` 在镜像里但不在 `PATH`，补上后 `import aiter` 成功（job 1123060）；
     - 门 2：`is_aiter_found_and_supported()` 末行是 `return on_mi3xx()`（job 1123259），
       而 `_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942","gfx950"])`（job 1123605）——
       BW1000 报 `gcnArchName='gfx936:sramecc+:xnack-'`，**恒为 False**，
       与 `rocminfo`、`PATH`、任何 `VLLM_ROCM_USE_AITER_*` 环境变量全部无关。
     → **加 PATH 不是修法，是白费。** 上游对 MI300 的硬编码判断，改它等于改 vLLM 源码。
  5. ~~换镜像到 0.21~~ **已排除**（加载权重即崩，见 §3.1 上方的修正）。
  6. **显式 `--attention-backend ROCM_AITER_MLA`** —— **❌ 已关闭**（job 1124165）。
     机制上确实能绕过 `on_mi3xx()`（`rocm.py:478` 显式分支直接 `get_class()` +
     `validate_configuration()`，不调 `_get_backend_priorities`），1 卡探测也拿到
     `get_class() -> AiterMLABackend`、`validate_configuration -> OK (empty)`。
     **但真实起引擎时四个 TP rank 同时报**：
     `ImportError: cannot import name 'flash_attn_varlen_func' from 'aiter'`。
     `validate_configuration()` 不导入模块，真 import 在初始化时——镜像的 aiter 版本
     没有 vLLM 0.20 `rocm_aiter_mla.py` 要的这个符号。`--enforce-eager` 同路径同失败。
     → **aiter 路线彻底关闭**，见 gotchas #49。不要再投卡时。
- **当前正在跑的三个方向**（2026-11-15 16:20 更新）：
  | 作业 | 方向 | 判据 |
  | --- | --- | --- |
  | ~~1124165 `kimi-aitermla`~~ | 显式 `ROCM_AITER_MLA` | ❌ **已关闭**：aiter 缺 `flash_attn_varlen_func` |
  | ~~1123874 `kimi-maxseq`~~ | 降批 24/16/32eager | ❌ **已关闭**：三档 round2 全崩 |
  | ~~1124366 `kimi-restart`~~ | 每轮重启服务，4×16K/c64 | ✅ **4/4 全过**（稳态 283.53–286.84 tok/s，各 128/128、0 VMFault）——**已收工** |
  | — | P0-B fp8 闸门 | ✅ 已结论：外部阻塞，停手 |
- **✅ P0-A 已达成（2026-11-15 17:27 终判，job 1124366）**：**每轮压测前重启服务进程**，
  16K/c64 **四轮全部通过**（各 128/128、0 处 VMFault）：

  | 轮 | 吞吐 | TTFT | TPOT | 备注 |
  | --- | ---: | ---: | ---: | --- |
  | R1 | 235.14 tok/s | 47.5 s | 218.3 ms | 冷启动，页缓存未暖，**不作稳态引用** |
  | R2 | 283.53 tok/s | 31.5 s | 187.4 ms | |
  | R3 | 284.62 tok/s | 30.8 s | 187.2 ms | |
  | R4 | **286.84 tok/s** | **30.6 s** | **185.7 ms** | |

  R2–R4 极差仅 **1.2%**，可直接当稳态数据引用。脚本自动判定原文：
  `[rs 17:27:57] 通过 4 / 4 轮 / 判定：H1 累计状态成立`。
  - **交付口径**：16K/c64 按"单轮 + 重启服务"出数，每轮多 3–4 分钟冷启动；
  - 这个数字**优于 `--enforce-eager`**（239.32 tok/s / 43.6 s），而且不需要任何降级开关；
  - 根因是**累计状态**（1120618 在约 1200 个请求后崩；1124366 每轮 128 个请求、进程活 27 分钟仍通过
    → 更像"累计请求数"而非"时钟时长"），
    不是 cudagraph、不是批大小、不是显存。详见 gotchas #51。
  - 四轮原始 jsonl 已归档到 `bench/restart-1124366/`（含 README 说明口径）；
    `bench/summary.md` 补了 16K/c64 行（取 R3 中位数 284.62），`meta.json` 的
    `throughput.16k_c64` 由 `null` 填为 284.62，`bench_note` 同步改为「6 档全部有效」。
    旧的「缺 paged-attn 内核」解释（gotchas #21）是**崩溃后的表现**，已在 summary 中更正。
- 成功判据：16K/c64 两轮都出数且 `mean_ttft_ms > 0`（用 `bench.sh` 的有效性校验）。
  ⚠️ **该有效性校验此前从未执行过**（heredoc 里两处 bug：`os` 未导入 + `ILENS` 未定义），
  已于 2026-11-15 修复，见 gotchas #45。**改完必须 `./tools/sync_remote.sh`**，
  否则远程 `zoo-sync` 的旧副本会被作业 rsync 进容器，校验段照样崩。
- ⚠️ **判定"跑完了"必须看 `Successful requests`，不能看进度条**（gotchas #50）。
  崩溃后失败请求会飞速把进度条刷到 100%。

**B. fp8 GEMM —— ✅ 已判定为外部阻塞（2026-11-15，作业 1123947，0 卡时）**

- **结论：DCU 的 torch 后端没有 fp8 GEMM 实现。**
  ```python
  torch.mm(torch.zeros(16,16,device='cuda',dtype=torch.float8_e4m3fn), ...)
  # NotImplementedError: "addmm_cuda" not implemented for 'Float8_e4m3fn'
  ```
  `float8_e5m2` 同样。这条一行命令就能复现，不需要加载模型。
- **四条旁证排除了所有"配置问题"解释**：
  | 看起来该有的 | 实际 |
  | --- | --- |
  | `fp8` 在量化白名单 | ✅ `RocmPlatform.supported_quantization` 含它 |
  | `Fp8Config` 可解析 | ✅ `get_quantization_config("fp8")` 有返回 |
  | `fp8_dtype` 有值 | ✅ `torch.float8_e4m3fn` |
  | hipBLASLt 认 fp8 | ✅ `libhipblaslt.so.0.10` 含 `R_8F_E4M3`/`R_8F_E5M2` |
  | torch 能分发 fp8 GEMM | ❌ **没有** |
- **为什么表现为"engine core 初始化失败"**：`get_device_capability()` 把 gfx936 硬套成
  `DeviceCapability(9, 3)`，vLLM 据此认为"fp8 应该支持"，一路放行到第一次预热 GEMM 才炸。
  报错位置离真因很远，这就是它一直被记成 gotchas #22 的原因。
- **与 A 的关系**：同一个根因的两面——aiter 的 fp8 kernel 也被 `on_mi3xx()` 挡在 gfx936 之外。
  但 attention 有"显式指定"这条出口（见 A-6），**GEMM 没有等价旋钮**。
- **动作**：**停手**。在 torch 侧补 fp8 GEMM 绑定之前，不自量化、不换版本、不再占卡时。
  完整证据：`sweeps/results-kimi-quantigate-2026-11-15.md`、gotchas #46。

**C. vLLM 长上下文整体（MiMo 16K 只有 SGLang 的 1/5）**

- 现状：MiMo × vLLM 16K/c8 = 44.83 vs SGLang 308.72（6.9×），但**能跑完**，属性能问题不是崩溃。
- 若 A 的 attention backend 结论可复用，这里可能顺带解决。

### 3.3 记录要求

每个子任务无论成败都要落三样：条目 README 的 §6/§7、`docs/gotchas.md` 新增条目（现象/判据/根因/解法），
以及 0 卡时的闸门命令与输出（`sweeps/<实验名>.md`）。

**压测数据的三条硬性判据**（2026-11-15 用一整天和 8 个作业换来的）：

1. **`Successful requests` 必须等于 `--num-prompts`** —— 进度条到 100% 完全不代表成功
   （崩溃后失败请求会飞速刷满，gotchas #50）。
2. **记录服务进程的已运行分钟数与累计成功请求数** —— 只记 `Running` 会误判，
   本仓库因此把"累计状态"误读成"批大小"问题，白跑 3 个作业（gotchas #51）。
3. **读日志前先核对 mtime** —— `bench/*.log` 不会自动清理，容易读到上一轮的残留（gotchas #47）。

---

## 4. P1：解除现有阻塞（"能做的都做"，每项 1–4 卡时）

| # | 阻塞 | 现状与证据 | 起手动作 | 判据 |
| --- | --- | --- | --- | --- |
| 1 | **Trident × SGLang**（优先级下调） | `Qwen3VLForEmbedding` 未注册（`ValueError: Cannot find model module`）。gotcha #17 实测：SGLang 0.5.10/0.5.18/main 的 `qwen3_vl.py` 均只注册 `ForConditionalGeneration`，无 embedding 变体、也没有 `qwen3_vl_embedding.py`。**已有可用替代**：自建 serving（见下）。 | 若要原生支持需给 SGLang 打源码补丁（非注册插件可解），本地无源码、无卡时，未验证——不要照旧描述写"注册插件"当作可交付。实际交付用 `../OpenBMB__Trident-Qwen3VL-2B__bw1000__transformers/` 的自建 serving | 服务能起，`/v1/embeddings` 返回 2048 维归一化向量（自建 serving 已达成，22.8 条/s @512 tok） |
| 2 | **Trident × vLLM**（优先级下调） | 同上；vLLM 0.20/0.21 均为架构级不支持。**旧方案有误**：本表原写"`--task embed`"，但该条目 README 实测 vLLM 0.20 `api_server` 报 `unrecognized arguments: --task`，嵌入模式由架构自动识别——`--task` 是新版写法，不适用 0.20。已有可用替代：自建 serving。 | 原生支持需给 vLLM 注册架构并改 ModelConfig 校验（框架补丁级），本地无法验证。**不要写 `--task embed`**。实际交付用自建 serving | 同上（自建 serving 已达成）；原生支持列 blocked 留证据 |
| 3 | ~~**Kimi 16K/c64**~~ ✅ 已解决 | 见 §3.2 A：2026-11-15 17:27 终判，每轮重启服务后四轮全过（128/128、0 VMFault，稳态 283–287 tok/s），交付口径"单轮+重启服务" | 无需再做；引用时按 §3.2 A 数字 | 已达成 |
| 4 | ~~**Qwen3-VL 类模型的 vLLM 带图崩溃**~~ ✅ 已解决（2026-11-21 复核） | vLLM 0.20 的 `deepstack tokens 72 > 67` 断言，**0.21 已删除该断言**（gotcha #27）。现有 RoboBrain/Recon2Reason × vLLM 0.21 条目的 `smoke.log` 实测 HTTP 200 + 纯文本/单图/多图（2 张）正确回答，`bench/summary.md` 6 档全绿。旧判据"4 图"查证为笔误：全仓无一处将 4 图作为该测试要求（其它"4 张"均指平台截图或 Kimi TP=4），T02 既定标准是 2 图 | 无需再做。若仍想扩模型，新建 Qwen3-VL-8B 条目属 §5（P2），不是本阻塞 | 带图（单图/2 图）请求 200 且无断言——已达成 |
| 5 | **OmniWeaving 8 卡 SP** | 4 卡已验证 2.23×；8 卡需 `-c 64 --mem≈480G`（本集群单节点 510 GB，需 `-c 120` 才够，可能卡 DefMemPerCPU） | 先 `-c 64 --mem=240G` 试 8 卡是否 OOM，再决定抬不抬内存 | 单条 < 662 s 才值得继续 |
| 6 | **Wan2.1 序列并行（ulysses）** ❌ 受阻（2026-11-21 实测） | FSDP 在 4 卡后饱和（8 卡仅 1.11×，单条 1003 s）。ulysses 依赖 xfuser，但 **xfuser 在当前 DCU 环境装不上**：作业 1187223（mimo-sglang 1.1.1 镜像，1 卡 4 min）实测——① 镜像内 `ModuleNotFoundError: No module named 'xfuser'`；② 计算节点 pip 源是内网 DTK 源 `10.16.1.201:9929/nightly/dtk2604/+simple/`，连接被拒（6 次重试失败），`No matching distribution found for xfuser`；③ 集群 wheels/softwares 目录均无 xfuser wheel | 不要再投卡时试 `import xfuser`——已确认装不上。除非后续能在登录节点下载 xfuser wheel 并挂进容器（但 xfuser 依赖 flash_attn，DCU 无可用构建，见 Wan README §8） | 受阻，标 blocked 留证据（作业 1187223 stdout 在 `luql:~/work/xfuser-probe/std.out.1187223`） |

**建议顺序**：5 → 1/2（第 6 项 Wan ulysses 已于 2026-11-21 实测受阻——xfuser 装不上，作业 1187223；第 3 项 Kimi 16K/c64、第 4 项 Qwen3-VL 带图均已解决；均已移出待办）。剩余 1/2 Trident 原生支持也已下调优先级（需框架补丁、本地无法验证、自建 serving 已跑通 22.8 条/s）。**实际上 P1 已无明显低风险可做项，下一步重心应转 §5（P2 扩模型）。**

---

## 5. P2：扩模型（每项 1–2 卡时，排在 kernel 之后）

| 模型 | 体积 | 已知情况 |
| --- | --- | --- |
| `Qwen/Qwen3-VL-8B-Instruct` | 17.5 GB | ModelScope 可得；vLLM 0.21 已修 deepstack 断言，是本仓库最省事的下一批 |
| `moonshotai/Kimi-VL-A3B-Thinking` | 32.8 GB | ModelScope 可得；需 2 卡或开启 offload |

新增条目照 `docs/entry-spec.md` 走（8 章 README + `env.sh` + `image.lock` + `bench/` + `logs/`），
跑完必须过四件套（见 §7）。

---

## 6. 阻塞与风险清单（不一定能解，但要有记录）

| 项 | 状态 | 解除条件 |
| --- | --- | --- |
| Trident × 两引擎 | blocked，已自建 serving 绕过（22.8 条/s，文本/图像/图文三类全通过） | 原生支持需框架补丁（gotcha #17），本地无法验证；不要写 `--task embed`（vLLM 0.20 无此参数）。交付以自建 serving 为准 |
| ~~Kimi 16K/c64~~ | ✅ 已解决（2026-11-15 17:27）：每轮重启服务，四轮全过（稳态 283–287 tok/s），交付口径"单轮+重启服务" | §3.2 A / gotcha #51 |
| fp8 / 量化 | 无 kernel | §3.2 B |
| T01/T02 精度 | **已查证：官方口径与我们的不可比，数据也取不到** | 不是"等材料"，也不是"取来就能跑"：官方分数是 8-shot/5-shot，我们是 0-shot；MiMo 根本没披露 MMLU/GSM8K；数据集本集群不可得。**建议不再投入**（§1-3b，`docs/t01-t02-baseline-investigation.md`） |
| Harbor 推送 | 平台无自助凭据 | 已决定不推，镜像留在 `luql` 家目录（tar.gz + sha256 兜底） |
| 平台清账号风险 | 3 个镜像源出第三方账号（`dcu/linshp`、`dcu/wqxyyt06`、`dcu/acizv7iztz`），上游可随时删 | tar.gz 归档是唯一保障；不要删镜像本体。2026-11-21 实测原 `archive/images/` 已不可见，现存镜像在 `scnethr2601a10/zoo-images/` 与 `luql/work/imgstore/`（见 §2.6），优先保留这些 |
| 多卡配额语义 | 每卡需 ≥8 CPU，否则 GRES 静默降级；`--mem ≤ -c × 3888M` | 已固化进各条目 `submit*.slurm` 注释 |

---

## 7. 环境与命令速查

```bash
# 账号（唯一可用）
ssh luql@zzeshell.scnet.cn -p 65032
# 旧账号（只能读写文件，不能提交作业）
ssh zzeshell-hz1                      # scnethr2601a10

# 归档
A=/public/home/luql/archive/zoo-dcu-2026-11
cd $A && sha256sum -c SHA256SUMS      # 11 条，全 OK 才可信

# 镜像还原（离线，比 pull 稳）
gunzip -c $A/images/<name>.tar.gz | podman --root /tmp/pr-$JOBID --runroot /tmp/prr-$JOBID \
  --storage-driver overlay --storage-opt ignore_chown_errors=true load

# 提交（每个条目都在自己目录里 sbatch；磁盘水位不够会 exit 42 自动重投）
cd ~/work/<条目目录> && sbatch container/submit.slurm

# 查作业
squeue -u luql -o "%.8i %.14j %.9T %.6M %.18R"
sacct -j <JID> -n -P -o JobID,State,ExitCode,Elapsed,AllocTRES%50,NodeList
```

**本机仓库与工具链**

```bash
cd ~/Downloads/agent/repo/dcu-model-test/zoo    # 2026-11-15 起的仓库位置（旧的 pi-test/zoo 已不存在）
./tools/lint_entry.sh          # 条目规范（8 章 README、image.lock、logs、bench 一致性）
python3 tools/gen_index.py     # 重生成 INDEX.md 与 README 总矩阵
./tools/check_quota.sh         # 配额/体积检查
python3 tools/check_products.py# 20 个条目产物与 stats 一致性
python3 tools/gen_report.py    # 重生成 docs/REPORT.md 的统计段
./tools/publish.sh "msg"       # commit + 推 GitHub（GitLab 远端已移除）
./tools/sync_remote.sh         # 把本地 models/ + sweeps/ 推到 luql:~/work/zoo-sync（提作业前必跑）
```

**工作纪律（上一段总结出来的，务必遵守）**

1. 四件套 + `publish.sh` 每轮都跑；数字口径靠脚本核，不靠人眼（`gen_report.py` 会打印条目/踩坑计数）。
2. **不许编造数据**：官方没下发的东西写 `[BLOCKED]`，不填估算值。
3. 跑不通的配置**标 blocked 并留证据**，不许删条目。
4. 每次踩坑立刻写 `docs/gotchas.md`（现象 / 判据 / 根因 / 解法 / 代价），一条不超 40 行。
5. 容器里的日志必须落宿主机（`podman logs` / 子进程日志目录），否则容器一删根因就没了。
6. 二进制只放 `archive/`，本机不留副本。

---

## 8. 新会话开场提示（可直接粘贴）

```
读 ~/Downloads/agent/repo/dcu-model-test/zoo/HANDOFF.md 与 docs/gotchas.md 最新条目（现 59 条）。

注意：§3.1 的 0 卡时闸门与 P0 已完成（2026-11-15）——闸门通过、P0-A（Kimi 16K/c64）用
"每轮重启服务"达成、P0-B（fp8）判定外部阻塞停手。不要再从头做 P0。

当前待办从 §4（P1）起，建议顺序 5 → 1/2：
- P1-5 OmniWeaving 8 卡（1–4 卡时；先 -c 64 --mem=240G 试是否 OOM）；
- P1-1/2 Trident 原生支持已下调优先级（需框架补丁、本地无法验证、自建 serving 已跑通 22.8 条/s），
  不要照旧描述写"注册插件"或 `--task embed`（vLLM 0.20 无此参数）。
- P1-6 Wan ulysses 已实测受阻（2026-11-21，作业 1187223）：xfuser 装不上（镜像无、计算节点 pip 源不可达、集群无 wheel），不要再投卡时试 import。
- **P1 已无明显低风险可做项；下一步重心转 §5（P2 扩模型：Qwen3-VL-8B / Kimi-VL）。**
- P1-4 Qwen3-VL 带图崩溃已解决（0.21 实测单图/2 图 200，旧"4 图"判据系笔误）；若想扩 Qwen3-VL-8B 见 §5（P2）。

资源现状（2026-11-21 实测）：hx1hdnormal 有 1 个空闲节点可提交；luql 家目录 85.1%（剩 217 GB，注意别撑爆）。
每个结论都要有原始日志证据；四件套 + tools/publish.sh 收尾（仅推 GitHub，GitLab 已移除）。
```
