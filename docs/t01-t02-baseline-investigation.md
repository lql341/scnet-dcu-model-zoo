# T01/T02 精度基线：官方澄清之后，到底还缺什么（2026-11-15 调查）

> **一句话结论**：官方「以公开基线为准、不再单独下发」这句话是**真的**，但它解开的只是
> 「配置文件由谁给」这一个结。**真正卡住 §6 的是评测口径与数据**——
> 官方公开的分数与我们 `verify.py` 采用的口径**不可比**，而对应的数据集在本集群**取不到**。
> 因此 §6 继续保持 `[BLOCKED]`，理由从"等官方发"改成"口径不可比 + 数据不可得"。

## 1. 调查起因

2026-11-15 用户转述官方澄清：**T01–T02 的验收标准用官方基线即可，不再提供单独的基线文件。**

此前的记录把这条当成"发个文件就能跑通"。本文件记录复核结果。

## 2. 查到了什么：官方公开分数（含出处）

### 2.1 Kimi-Linear-48B-A3B-Instruct

来源：Kimi Linear 技术报告 arXiv 2510.26692，**Table 9（Instruct 模型，5.7T tokens）**。

| 指标 | Kimi-Linear-Instruct | 口径 |
| --- | ---: | --- |
| MMLU-Redux（EM） | 86.9 | 未在表中标注 shot 数 |
| MMLU-Pro（EM） | 72.7 | 同上 |
| GPQA-Diamond | 71.7 | Avg@8 |
| MATH500 | 94.6 | Acc. |
| LiveCodeBench v6 | 45.7 | Pass@1 |

**关键缺口**：Table 9 里**没有 plain MMLU，也没有 GSM8K**。

GSM8K 数字出现在 **Table 8（base 模型）**：

| 指标 | Kimi-Linear-**Base** | 口径 |
| --- | ---: | --- |
| GSM8K | 86.3 | **8-shot** |
| GSM8K-platinum | 89.6 | **8-shot** |
| MMLU-Pro | 54.8 | 5-shot |
| MMLU-Redux | 79.7 | 5-shot |

**即：GSM8K 86.3 是 base 模型 8-shot 的成绩，不是 instruct 模型的。**
拿它当 instruct 的验收基线，等于在比较两个不同模型。

### 2.2 MiMo-V2.6-Distill-Qwen-9B

来源：官方模型卡 + MiMo-V2.6 技术报告（arXiv 2610.11959）。

**没有任何 MMLU 或 GSM8K 数字。** 官方报告的全部指标都是 agentic / coding 类：

| 指标 | SFT 版 |
| --- | ---: |
| SWE-bench Verified | 61.1（avg@3） |
| SWE-bench Pro | 44.6（avg@3） |
| Terminal Bench 2.1 | 37.1（avg@1） |
| Toolathlon-Verified | 35.2（avg@1） |
| JobBench | 18.3（avg@1） |

官方明确把它定位为 agentic 方向的蒸馏模型，主体指标在 SWE / Terminal / Tool 一类。
**它根本没有我们 `verify.py` 要评的那两个榜。**

## 3. 为什么这不能直接填进 `baselines/`

### 3.1 口径不可比

`verify.py` 的口径（写在脚本里，2026-11-01 建库时定的）：

| 项 | 我们的口径 |
| --- | --- |
| 采样 | **greedy（temperature=0）、0-shot** |
| GSM8K 判分 | 取 `\boxed{}` 或末尾数字，与答案数值相等 |
| MMLU 判分 | 取首个 A–D 字母 |
| 容差 | 劣化 ≤ 2 pt |

官方口径（上面查到的）：**8-shot**（GSM8K）/ **5-shot**（MMLU）/ **EM**（Redux/Pro）。

**0-shot instruct 成绩去比 8-shot base 成绩，+2pt 的容差没有意义**——
差出来的每一分都可能是 shot 数造成的，而不是硬件或框架造成的。
这恰恰是 `docs/entry-spec.md` §6 红线 1 要防的事：**看起来通过了验收，实际不可比**。

### 3.2 `verify.py` 的评测配置本来就没有出处

复核 git 历史（`f7e1806`，2026-11-01 建库）后确认：
`verify.py` 里"从 `baselines/<模型>.json` 读取子集路径/题数/max_tokens/基线分"这套设计，
是在**假设**官方会下发该文件的前提下写的。**该文件从未存在过**，
子集与题数也从未由官方定义。

所以"取到官方数值按原格式落盘即可跑通"这个说法**不成立**——
即使拿到分数，`n` 填多少、`subset` 指哪个文件、`max_tokens` 用什么，仍然没有官方依据。

### 3.3 数据集也取不到

| 通道 | 状态 |
| --- | --- |
| huggingface.co | `Connection timed out` |
| github.com | `Connection timed out` |
| www.modelscope.cn | 可达（302），但 `modelscope/gsm8k` 仓库只有 `gsm8k.py` 加载脚本 + README，**没有数据文件**；该脚本会去 OpenAI 的 CDN 取数据，同样被墙 |

本集群 **拿不到 GSM8K / MMLU 的题目本体**。

## 4. 结论与处置

**§6 继续 `[BLOCKED]`，但理由变了**：

| | 旧理由 | 新理由 |
| --- | --- | --- |
| 性质 | 等官方下发文件 | **官方公开分数与我们的评测口径不可比；且数据集本集群不可得** |
| 谁能解 | 官方 | 需先定评测口径，再找到可用的数据通道 |
| 是否值得解 | 等 | 见下 |

**为什么不再花力气去解**（用户 2026-11-15 决策：项目性质是能力沉淀，不是比赛提交）：

1. 即便凑出一套自测数字，它也**不能用来判定"劣化 ≤2pt"**——口径不同，比较无效；
2. 硬做只会产出"看起来像验收通过"的数字，这正是红线要防的；
3. 本仓库的 §6 空白**是如实记录，不是缺失**。

**如果将来确实要补**，正确顺序是：

1. 先确定一个**书面口径**（几 shot、哪个子集、max_tokens），并记录出处；
2. 找一个**本集群可达**的数据通道（modelscope 其他镜像仓库 / 平台附件 / 手工上传）；
3. 只在这两个前提都成立后，才谈"跑分"；且结果必须标注**评测口径**，
   不得与官方数字直接相减求"劣化"。

## 5. 各条目的现状

| 条目 | §6 状态 | 说明 |
| --- | --- | --- |
| Kimi × SGLang / vLLM | `[BLOCKED]` | 官方有 MMLU-Redux/Pro，但口径不明且无 plain MMLU/GSM8K |
| MiMo × SGLang / vLLM | `[BLOCKED]` | 官方公开指标里完全没有 MMLU/GSM8K |
| Recon2Reason × 两引擎 | 豁免 | 赛题 §五 明确"可以不做"；以多模态链路 smoke 为证据 |
| RoboBrain × 两引擎 | 豁免 | 同上 |
| Trident × 三引擎 | 不适用 | 架构不被 SGLang/vLLM 支持，已另建 serving |

## 6. 复核方式

本文所有官方数字均可在下面两处独立核对：

- Kimi Linear 技术报告 `arxiv.org/pdf/2510.26692`：Table 8（base，含 GSM8K 8-shot 86.3）、
  Table 9（instruct，仅 MMLU-Redux/Pro，无 GSM8K）
- MiMo-V2.6 技术报告 `arxiv.org/html/2610.11959v1`：Table 6（distill 的 agentic 指标），
  全文无 MMLU / GSM8K

复核日期：2026-11-15。
