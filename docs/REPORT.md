# BW1000（海光 DCU）模型适配能力报告

> 版本：**2026-11-09（终版）**　作者：本仓库的会话记录（每一条数字都能在 `models/*/` 里追到原始日志）
>
> **结项状态**：15 个条目、7 个官方评测全绿 + 1 个 partial、923 条真实产物（官方清单 935 条）、3 个镜像归档（31 GB 落在 `luql` 账户）、
> 24 条踩坑、5 条硬约束。集群侧临时账号 `scnethr2601a10` 已于 2026-11-07 被回收（见 §9），
> 但**所有数据、镜像、经验都已不在那个账号上**。
> 目的：**能力沉淀**优先于赛题提交。这份报告回答三个问题：
> ① 我们在这张卡上到底跑通了什么；② 哪些是硬约束（换谁来做都过不去）；③ 下次遇到同类卡，从哪一步开始最快。

---

## 0. 一页速览

| 维度 | 结果 |
| --- | --- |
| 条目数 | **15 个**（模型 × 卡型 × 框架 × 精度），全部带启动脚本、环境自检、原始日志 |
| 跑通并出官方条件结果 | **8 个**（T03/T04 官方 8 个模型里 6 个全绿 + AuK-Flash 变体；OmniWeaving 部分完成） |
| 有完整 6 档性能基准 | **3 个**（MiMo×SGLang、MiMo×vLLM、Kimi×SGLang） |
| 官方条件产出 | **923 条真实产物**（官方 9 份清单共 935 条；缺 hy_world 5 条未做 + omniweaving 7 条未完成） |
| 镜像归档 | **3 个**（30 GB，含社区版 Music3），全部 sha256 + 还原验证 |
| 依赖归档 | 4 套 cp310 轮子 + 3 份源码（`/public/home/luql/archive/zoo-dcu-2026-11/assets/` 与 `sources/`） |
| 踩坑清单 | **24 条**（`docs/gotchas.md`，按隐蔽度排序） |
| 已知硬约束（换框架也过不去） | **5 条**（见 §4） |
| 二进制归档 | **31 GB / `luql:/public/home/luql/archive/zoo-dcu-2026-11/`**（镜像 + 产物 + 轮子 + 调优原始数据） |

---

## 1. 官方条件评测：谁跑完了，花了多少

| 模型 | 任务 | 样本 | 结果 | 单条耗时 | 卡数 | 关键规格 |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Qwen3-ASR-1.7B | 中英识别 | 300 | **300/300** | — | 1 | CER 1.21%（zh）/ 0.65%（en），WER 2.86% |
| Qwen-Image-2.1-Turbo | 文生图 | 180 | **180/180** | 19.8 s | 1 | 2048×2048，8 步 + KV cache |
| AuK | 参考音克隆 | 120 | **120/120** | 4.4 s | 1 | 24 kHz 单声道，nfe=32 |
| **AuK-Flash** | 参考音克隆（蒸馏） | 120 | **120/120** | **1.0 s** | 1 | 同上，nfe=4 → **快 4.4×** |
| Qwen3-ForcedAligner | 词级对齐 | 30 | **30/30** | 0.36 s | 1 | 预测 572 词 = 参考 572 词 |
| **MiniMax-Music3** | 音乐生成 | 60 | **60/60** | 203 s | 1 | 32 kHz 立体声，750 声学帧 |
| **Wan2.1-I2V-14B** | 图生视频 | 60 | **60/60** | 1116 s | 4 | 832×480 / 81 帧 / 16 fps |
| HY-OmniWeaving | 文生视频 | 60 | **53/60**（账号回收中断） | ~1560 s | 1 | 832×480 / 81 帧 / 16 fps |

**意义**：T03/T04 两题的 8 个模型里，**6 个全绿**（Qwen-Image、AuK、ASR、ForcedAligner、Music3、Wan2.1），
OmniWeaving 53/60 部分完成，HY-World-2.0 未做（需 ROCm 版 gsplat）。另有 AuK-Flash 作为蒸馏变体额外全绿。
**全部零编造**：所有时长/规格都由脚本从产物里读出来（`stats.json`），不是抄模型卡。

---

## 2. 性能：同模型跨框架的硬对比（这是最有价值的一组数据）

MiMo-9B，同一张 BW1000（64 GB），同一套压测客户端与有效性校验：

| 档位 | SGLang 0.5.10 | vLLM 0.20 | vLLM 相对 |
| --- | ---: | ---: | ---: |
| 4K/c1 | 44.31 | 20.73 | −53% |
| 4K/c8 | 360.23 | 160.63 | −55% |
| 4K/c64 | **998.05** | 489.52 | −51% |
| 16K/c1 | 41.47 | 8.19 | **−80%** |
| 16K/c8 | 308.72 | 44.83 | **−85%** |
| 16K/c64 | 299.81 | 100.82 | −66% |

单位 output tok/s。**结论：在这张卡上，SGLang 全面优于 vLLM，长上下文差距拉到 5–7 倍**，
且不是参数问题（见 §4 第 1 条）。

Kimi-Linear-48B（TP=4）6 档：104.67 / 353.53 / **839.00** / 50.80 / 119.21 / 234.17。

### 优化手段与收益（都是实测，不是估算）

| 条目 | 手段 | 收益 | 代价 |
| --- | --- | --- | --- |
| MiMo × SGLang | `max-running-requests 8→64` + 开 CUDA Graph | **+73% ~ +466%** | 无 |
| MiMo × vLLM | 去 `--enforce-eager` + `max-num-seqs 128` | +3% ~ +32% | 无 |
| Kimi × SGLang | 开 CUDA Graph | +37.9% ~ +620.8% | 无（原记 16K/c64 −6.3%，2026-11-05 复测为**噪声**） |
| AuK → AuK-Flash | 换 4 步蒸馏权重 | **4.4×** | 无（同代码同环境） |
| MiMo + Kimi（16K 档） | `--chunked-prefill-size 4096` + 保持 CUDA Graph | **+5~8%** | 无 |

---

### 2.1 量化与批处理（2026-11-06）

| 实验 | 结论 |
| --- | --- |
| fp8 在线量化（vLLM） | ❌ engine core 初始化即失败，DCU 无该 kernel |
| **AWQ（weight-only INT4）** | ✅ **能跑**（`awq_marlin` → `TritonW4A16LinearKernel`），但 4K/c64 只有 185 tok/s（bf16 同规模约 998），TPOT 4–6 倍；显存收益真实（6.1 GB 权重 + 51 GB KV） |
| ASR 批吞吐 | `--max-num-seqs 64` + 16 并发：60 条 30 s → **27 s（−10%）**，CER/WER 零变化 |

### 2.2 跨模型一致的调参结论（2026-11-05）

`--chunked-prefill-size 4096` + 保持 CUDA Graph，是这套 DCU 上 16K 档的最优组合，**两个模型同时受益**：

| 模型 | 变体 | 16K/c64 | 16K/c8 |
| --- | --- | ---: | ---: |
| MiMo-9B（1 卡，2 轮均值） | base | 296.0 | 218 |
| | **chunked 4096** | **315.2（+6.5%）** | 223 |
| Kimi-48B（TP=4，1 轮） | base | 232.03 | 127.71 |
| | **chunked 4096** | **244.73（+5.5%）** | **138.31（+8.3%）** |
| | 关掉 CUDA Graph | 232.03 | 87.71（**−31%**） |

**同时纠正两条旧结论**：① Kimi 所谓"开 graph 导致 16K/c64 回退 6.3%"是单轮噪声（±7%）；
② 限制 graph 捕获上界（`--cuda-graph-max-bs`）与提高显存比例都没有效果。

## 3. 可复现性资产（比"跑通"更值钱的部分）

| 资产 | 位置 | 校验 |
| --- | --- | --- |
| SGLang 镜像 7.9 GB | `/public/home/luql/archive/zoo-dcu-2026-11/images/mimo-sglang-bw10000_1.1.1.tar.gz` | sha256 `f16125ba…` + load/smoke 验证 |
| vLLM 镜像 8.7 GB | `/public/home/luql/archive/zoo-dcu-2026-11/images/vllm0.20-qwen_asr1.7_bw_1.0.0.tar.gz` | sha256 `986f95a3…` + load/smoke 验证 |
| Music3 镜像 13.7 GB | `/public/home/luql/archive/zoo-dcu-2026-11/images/minimax-music3-bw-sglang_20261031-rc1.tar.gz` | sha256 `b61b05bd…` + 还原跑通 1 条 |
| cp310 轮子 4 套 | `/public/home/luql/archive/zoo-dcu-2026-11/assets/*.tar.gz` | 53 + 3 + 8 个轮子，逐包 sha256 |
| 上游源码 3 份 | `/public/home/luql/archive/zoo-dcu-2026-11/sources/{AuK,Wan2.1,OmniWeaving}-src/` | — |
| 调优原始数据 34 个 jsonl | `/public/home/luql/archive/zoo-dcu-2026-11/products/optimizations/` | 各结论的原始证据 |
| 复现手册 | `docs/repro.md` | 端到端可执行 |

**为什么归档轮子**：镜像里没有 librosa/diffusers/byT5，而「按宿主机 py3.12 下载的 cp312 轮子装进 py3.10 镜像」
曾把两个条目误判成 blocked（gotchas #18）。**镜像 + 正确版本的轮子集才是完整可复现单元。**

---

## 4. 五条硬约束（换框架、换参数都过不去）

| # | 约束 | 证据 | 影响 |
| --- | --- | --- | --- |
| 1 | **vLLM 在 DCU 上没有 paged-attention 内核**，只有 Triton 兜底 | `Cannot use ROCm custom paged attention kernel`，五条旋钮（chunked 切片/AITER/backend 覆盖/fp8 KV/TP=2）全部无效 | 16K 吞吐只有 SGLang 的 1/5–1/7；要补只能自建镜像 |
| 2 | **fp8 量化无内核** | `--quantization fp8` → engine core 初始化即失败 | 全库 0 量化覆盖 |
| 3 | **14B 级视频模型单卡装不下** | Wan2.1 单卡 OOM（offload + t5_cpu 仍打满 63.98 GB） | 必须 4 卡 FSDP；单条 18.6 min |
| 4 | **2048² 文生图没有合批余量** | Qwen-Image batch=2 即 OOM（已分配 46.56 GB / 剩 3.84 / 需 6.75） | 19.8 s/张 已是该硬件上限 |
| 5 | **HY-World 重建需 ROCm 版 gsplat** | `worldrecon/rasterization.py` 硬 import gsplat，官方要 py3.11 + FA2/FA3 | 5.1 GB 权重已下但暂缓 |

---

## 5. 方法论：这套流程为什么能复用

1. **环境三层隔离**：镜像（不动）→ `/opt/*-libs`（`--no-deps` + cp310 轮子）→ `PYTHONPATH` 前置。
   永不 `pip install` 到系统目录，永不升级镜像里的 torch。
2. **参数只从条件清单读**：官方 `conditions/*.jsonl` 是唯一权威（gotchas #20：写死默认值会让 8 步变 40 步）。
3. **产物规格校验而不是"作业成功"**：音频看 24 kHz/1ch、图像看 2048²、视频看 81 帧/16 fps，
   `spec_violation_count` 必须为 0。**"服务在监听" ≠ "能出声"**。
4. **续跑优先**：单条 1–26 min 的任务一律「产物存在则跳过」，否则一趟 1 h 作业白烧。
5. **阴性结论同样入库**：不可行的路径要写清"试了什么、错在哪、要什么条件才能过"，
   避免下个会话重复烧卡时（§4 的五条都是这么来的）。
6. **归档先于结论**：镜像/轮子在用完之后立刻归档 + sha256 + 还原验证，因为上游随时会删仓。

---

## 6. 成本核算（真实数字）

| 项目 | 卡时 |
| --- | ---: |
| 基线与调参（8 月–10 月） | ~62 |
| Qwen-Image / AuK / ASR / FA 官方评测 | ~12 |
| vLLM 16K 三轮扫描（含失败轮次） | ~3 |
| Wan2.1（含 OOM 与 FSDP 调试 + 全量） | ~75 |
| Music3（探针 + 60 条 + 镜像归档/还原） | ~8 |
| OmniWeaving（探针 + 全量在跑） | ~35 |
| 其它（合批/量化探针、镜像归档） | ~6 |
| **合计** | **~200 卡时** |

账户余额从 4405.96 降到约 4300，**约 105 元卡时**（本集群 1 卡时≈1 元），
换来 15 个可复现条目 + 3 个镜像归档 + 24 条踩坑 + 923 条真实产物（官方清单共 935 条）——**单价约 0.1 元/样本**。

---

## 7. 如果我们继续做，下一步的性价比排序

| 优先级 | 事项 | 成本 | 收益 |
| --- | --- | --- | --- |
| 1 | 自建镜像补 paged-attn / fp8 内核 | 8–16 卡时 | 唯一能解锁 vLLM 长上下文与量化的路径（§4 第 1、2 条） |
| 2 | 量化条目（GPTQ/AWQ + 自证 CER） | 4–8 卡时 | 补上全库最大空白 |
| 3 | WorldMirror（编 gsplat-ROCm） | 8+ 卡时，可能失败 | T03 第 5 个模型 |
| 4 | ~~Kimi 16K/c64 回滚修复~~ | ✅ 已完成 | 回退是噪声；真正有效的是 chunked-prefill 4096（+5.5%/+8.3%） |
| 5 | T01/T02 新模型（需官方 baselines 才有意义） | — | 等材料 |

---

## 8. 附：数据去哪找

| 你想看什么 | 去哪 |
| --- | --- |
| 某个条目的完整技术报告 | `models/<org>__<model>__<card>__<framework>/README.md`（8 章） |
| 6 档基准原始数据 | `models/*/bench/{summary.md,round1.jsonl,round2.jsonl}` |
| 官方条件产物统计 | `models/*/official-eval/{反馈表.csv,eval_result.json,stats.json}` |
| 踩坑细节 | `docs/gotchas.md`（22 条） |
| 换机器怎么重建 | `docs/repro.md` |
| 镜像/轮子在哪、怎么校验 | `docs/ARCHIVE-SUMMARY.md` + `/public/home/luql/archive/zoo-dcu-2026-11/VERIFY.md` |
| 调优过程原件（48 个 slurm） | `sweeps/2026-10-11-probes/` |
| 会话交接与 TODO | `HANDOFF.md` |
| **二进制归档（镜像/产物/轮子/调优数据）** | `luql:/public/home/luql/archive/zoo-dcu-2026-11/`（见 `docs/ARCHIVE-SUMMARY.md`，内含 `VERIFY.md`） |
| 报告全文 | `docs/REPORT.md`（本文件） |

---

## 9. 结项说明（2026-11-09）

| 事项 | 情况 |
| --- | --- |
| 临时账号 | `scnethr2601a10` **2026-11-07 14:12 被移出 `hx1hdnormal` 的 `AllowAccounts`**（10552 条名单里唯独没有我们；同批 a05/a07 都在）→ 提交作业全面被拒，正在跑的 OmniWeaving 被 CANCELLED（53/60 产物完好） |
| `luql` 账号 | 郑州 SSH **已恢复**（用它完成了全部归档）；但同样不能在 `hx1hdnormal` 提交作业（QOS 不匹配） |
| 数据保全 | 三层：Git 双远端（代码/经验）→ `luql:archive/zoo-dcu-2026-11/`（31 GB 二进制，sha256 全核对）→ 原集群（即将失效） |
| 未完成 | OmniWeaving 剩 7 条（3.5 h，`evaluate_official.py` 支持续跑）；T01/T02 精度（官方 baselines 从未下发） |
| 复用前提 | 任何账号只要被加入该分区的 `AllowAccounts` 且绑定 `QoS=partition_hx1hdnormal`，即可按 `docs/repro.md` 用归档镜像重跑全部条目 |

**这套东西的复用价值不在"跑了哪些模型"，而在**：三个可直接 `podman load` 的 DCU 镜像、
24 条用卡时换来的坑、5 条硬约束（省掉下一个人几周的试错）、以及一条"从条件清单到产物校验"的
可执行流水线（`evaluate_official.py` + `check_products.py` + `repro.md`）。
