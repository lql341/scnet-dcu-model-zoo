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
| 踩坑 | **36 条**（`docs/gotchas.md`，按隐蔽度排序） |
| 归档 | 5 镜像 55 GB + assets + sources + **products 10 个目录**，`SHA256SUMS` 11/11 通过 |
| 代码同步 | GitHub `lql341/scnet-dcu-model-zoo` = GitLab `luql/scnet-dcu-model-zoo` = 本地（每次改完 `tools/publish.sh`） |

**本段最重要的三个教训**（写在这里，防止新会话重犯）：

1. **先看调度器的 `AllocTRES`，再看容器，最后看 torch** —— `--gres=dcu:N` 可能被静默降级成 1 张卡，
   每张 DCU 需 ≥8 CPU（`-c 64` 配 8 卡）。见 gotchas #30/#31。
2. **`2>/dev/null` 会把"拷仓库失败"变成"缺 Python 包"** —— `WORK` 与 `REPO` 相同会让 `cp` 把自己拷进自己，
   于是 `patches/` 没进容器、报出完全无关的 `ModuleNotFoundError: dashscope`。见 gotchas #33。
3. **报告失败前先确认"你测的是系统行为还是自己的超时"** —— 压测客户端默认只等 60 s，
   而 48B 模型冷启动 410 s；不改超时永远看不到真因（GPU VMFault）。见 gotchas #36。

---

## 1. 本轮决策记录（2026-11-15，用户拍板）

| # | 决策 | 影响 / 落地方式 |
| --- | --- | --- |
| 1 | **投入 kernel 主线** | 见 §3（P0），这是唯一能同时解锁"vLLM 长上下文"和"量化"的路径 |
| 2 | 量化暂不做 | 见 `docs/quantization-branch.md`：AWQ 慢 5×、fp8 无 kernel，先补 kernel 才有意义 |
| 3 | T01/T02 官方材料不管了 | 精度自评保持 `[BLOCKED]`，不填估算值；Recon2Reason/RoboBrain 维持"链路 + 基准"口径 |
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

### 3.2 三个子目标（按证据强度排序，建议这个顺序）

**A. Kimi-Linear 长上下文崩溃（证据最硬，优先级最高）**

- 现象（2026-11-15 实测）：Kimi × vLLM，16K/c64 时 worker 被 GPU 访存错误打死：
  `Invalid address access: 0x7f7e01015000, Error code: 1` / `KERNEL VMFault` /
  `Worker proc VllmWorker-3 died unexpectedly`；崩溃时 `num_running_reqs=33`，**无 OOM**。
  同配置 16K/c1、c8 完全正常（34.78 / 184.51 tok/s）。
- **强烈线索（不要重复上一段"缺 paged-attn"的错误判断）**：从 vLLM 启动配置的 `splitting_ops` 看，
  Kimi-Linear 走的是 `short_conv` / `linear_attention` / `gdn_attention_core` 这类**线性注意力**路径，
  与 paged-attention 无关。`fla_core`（flash-linear-attention）轮子存在（`~/wheels` 里有 0.6.0）。
- 起手动作（按代价从低到高）：
  1. 换 `--attention-backend`（TRITON_ATTN / FLEX_ATTENTION / ROCM 系）各跑一次 16K/c64；
  2. `--max-num-seqs` 保持 64 不变，改 `--max-model-len` 17408（=16K+1K）看是否只是缓冲越界；
  3. 若定位到 fla_core 的 fused_recurrent/chunk kernel，再评估是"换版本"还是"补 kernel"。
- 成功判据：16K/c64 两轮都出数且 `mean_ttft_ms > 0`（用 `bench.sh` 的有效性校验）。

**B. fp8 GEMM（解锁量化的前提，**只有 A 完成后才值得做**）**

- 现状：`--quantization fp8` 在 DCU 上 engine core 初始化即失败（gotchas #22）。
- 目标：让 Qwen3-ASR 的 fp8 子集能起服务；或证明 hipBLASLt 的 fp8 在 gfx936 上不可用并留证据。

**C. vLLM 长上下文整体（MiMo 16K 只有 SGLang 的 1/5）**

- 现状：MiMo × vLLM 16K/c8 = 44.83 vs SGLang 308.72（6.9×），但**能跑完**，属性能问题不是崩溃。
- 若 A 的 attention backend 结论可复用，这里可能顺带解决。

### 3.3 记录要求

每个子任务无论成败都要落三样：条目 README 的 §6/§7、`docs/gotchas.md` 新增条目（现象/判据/根因/解法），
以及 0 卡时的闸门命令与输出（`sweeps/<实验名>.md`）。

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
| T01/T02 精度 | 官方 `baselines/` 与评测器从未下发 | 已决定不管（§1-3） |
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
cd ~/Downloads/pi-test/zoo
./tools/lint_entry.sh          # 条目规范（8 章 README、image.lock、logs、bench 一致性）
python3 tools/gen_index.py     # 重生成 INDEX.md 与 README 总矩阵
./tools/check_quota.sh         # 配额/体积检查
python3 tools/check_products.py# 20 个条目产物与 stats 一致性
python3 tools/gen_report.py    # 重生成 docs/REPORT.md 的统计段
./tools/publish.sh "msg"       # commit + 推双远端（GitHub + GitLab）
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
读 ~/Downloads/pi-test/zoo/HANDOFF.md 与 docs/gotchas.md 最后 6 条（#31–#36），
然后从 §3.1 的 0 卡时闸门开始：在 1 卡作业里检查 vLLM 镜像的 attention backend 列表、
aiter 是否存在、hipBLASLt 的 fp8 支持、以及 Kimi-Linear 走的是哪套 kernel。
闸门结果决定 P0 是否继续投卡时。四件套 + publish.sh 收尾，每个结论都要有原始日志证据。
```
