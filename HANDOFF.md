# HANDOFF —— 下一段推进的交接文档

> 版本：2026-11-15　**本文件是唯一有效的交接文档**（旧的 2026-11-14 版已删除，历史细节见 `docs/ARCHIVE-SUMMARY.md` 与 `docs/REPORT.md`）
> 项目性质：**能力沉淀，不是比赛提交**（2026-11-15 决策）。所有产出的第一价值是"下一个人能复用"。
> 唯一二进制归档：`luql:/public/home/luql/archive/zoo-dcu-2026-11/`（55 GB）

---

## 0. 三分钟上手

| 项 | 现状 |
| --- | --- |
| 条目 | **20 个**（14 个不同模型 × 卡型 × 框架 × 精度） |
| 官方评测 | **9 个条目全绿，935/935 条真实产物**（官方 9 份清单 9/9 覆盖） |
| 跨引擎基准 | **8 个条目**（MiMo / Kimi / Recon2Reason / RoboBrain，各 SGLang + vLLM × 6 档） |
| 自建运行时 | **1 个**（Trident-Qwen3VL-2B：OpenAI 兼容 embedding 服务，两引擎均不支持该架构） |
| 阻塞 | **2 个**（Trident × SGLang/vLLM 架构未注册，保留证据） |
| 踩坑 | **51 条**（`docs/gotchas.md`，按隐蔽度排序） |
| 归档 | 5 镜像 55 GB + assets + sources + **products 10 个目录**，`SHA256SUMS` 11/11 通过 |
| 代码同步 | GitHub `lql341/scnet-dcu-model-zoo` = GitLab `luql/scnet-dcu-model-zoo` = 本地（每次改完 `tools/publish.sh`） |

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
| 经验 | 36 条踩坑 + 4 条有效硬约束（1 条已推翻） | `docs/gotchas.md`、`docs/REPORT.md` §4 |

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

## 3. P0：kernel 主线（唯一技术主线，预算 8–16 卡时）

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
| 1 | **Trident × SGLang** | `Qwen3VLForEmbedding` 未注册（`ValueError: Cannot find model module`） | 写注册插件：`SGLang.ModelRegistry.register("Qwen3VLForEmbedding", cls)`（cls 用 `trident_embedder._embedding_backbone_cls()` 现拼），用 `--model-type` 或 sitecustomize 注入 | 服务能起，`/v1/embeddings` 返回 2048 维归一化向量 |
| 2 | **Trident × vLLM** | 同上（vLLM 0.20/0.21 都没有该 arch） | `ModelRegistry.register_model("Qwen3VLForEmbedding", cls)`；注意它是 embedding 模型，需要 `--task embed` 与 `PoolerConfig` | 同上 |
| 3 | **Kimi 16K/c64** | 见 §3.2 A | 先做 §3.1 闸门 | 两轮出数 |
| 4 | **Qwen3-VL 类模型的 vLLM 带图崩溃** | vLLM 0.20 的 `deepstack tokens 72 > 67` 断言；**0.21 镜像已修**（Recon2Reason/RoboBrain 已在用 0.21） | 新增 `Qwen/Qwen3-VL-8B-Instruct × {SGLang, vLLM 0.21}` 条目，或对现有 0.21 条目补一次"多图 4 张"的 smoke | 带图 4 图请求 200 且无断言 |
| 5 | **OmniWeaving 8 卡 SP** | 4 卡已验证 2.23×；8 卡需 `-c 64 --mem≈480G`（本集群单节点 510 GB，需 `-c 120` 才够，可能卡 DefMemPerCPU） | 先 `-c 64 --mem=240G` 试 8 卡是否 OOM，再决定抬不抬内存 | 单条 < 662 s 才值得继续 |
| 6 | **Wan2.1 序列并行（ulysses）** | FSDP 在 4 卡后饱和（8 卡仅 1.11×） | 验证 xfuser 在 DCU 可用性（`import xfuser`），再试 `--ulysses_size 4/8` | 单条 < 1003 s |

**建议顺序**：4 → 1/2（Trident 插件是纯软件，收益最确定）→ 3 → 6 → 5。

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
| Trident × 两引擎 | blocked，已自建 serving 绕过 | §4-1/§4-2 的注册插件 |
| Kimi 16K/c64 | 崩溃（GPU VMFault），有原始日志 | §3.2 A |
| fp8 / 量化 | 无 kernel | §3.2 B |
| T01/T02 精度 | **已查证：官方口径与我们的不可比，数据也取不到** | 不是"等材料"，也不是"取来就能跑"：官方分数是 8-shot/5-shot，我们是 0-shot；MiMo 根本没披露 MMLU/GSM8K；数据集本集群不可得。**建议不再投入**（§1-3b，`docs/t01-t02-baseline-investigation.md`） |
| Harbor 推送 | 平台无自助凭据 | 已决定不推，镜像留在 `luql` 家目录（tar.gz + sha256 兜底） |
| 平台清账号风险 | 3 个镜像源出第三方账号（`dcu/linshp`、`dcu/wqxyyt06`、`dcu/acizv7iztz`），上游可随时删 | tar.gz 归档是唯一保障；不要删 `archive/images/` |
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
./tools/publish.sh "msg"       # commit + 推双远端（GitHub + GitLab）
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
读 ~/Downloads/agent/repo/dcu-model-test/zoo/HANDOFF.md 与 docs/gotchas.md 最后 6 条（#31–#36），
然后从 §3.1 的 0 卡时闸门开始：在 1 卡作业里检查 vLLM 镜像的 attention backend 列表、
aiter 是否存在、hipBLASLt 的 fp8 支持、以及 Kimi-Linear 走的是哪套 kernel。
闸门结果决定 P0 是否继续投卡时。四件套 + publish.sh 收尾，每个结论都要有原始日志证据。
```
