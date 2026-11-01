# 官方尚未下发的材料 —— 阻塞清单

本文件记录「按官方交付物要求，本仓库还缺什么、缺的东西归谁、影响哪一章」。
截止：2026-10-29。

## 一句话结论

15 项交付物中 **12 项已完成或可自行完成**；剩下 3 项全部阻塞在**官方未下发材料**上，其中 **1 项会直接导致验收不通过**（精度 Gate）。

## 阻塞项

### ① `baselines/<模型>.json` — 🔴 阻塞验收，只能官方给

- **要求出处**：《赛事详情》§六 仓库目录；模板 §2「评测配置与官方 GPU 基线写在官方下发的 `baselines/<模型>.json` 中，**不需要也不允许修改**」
- **本件需要的文件名**：`baselines/moonshotai__Kimi-Linear-48B-A3B-Instruct.json`
- **内容**：`subset` 路径、`n` 题数、`max_tokens`、思考开关（`extra_body`）、`ref` 官方 GPU 基线分数、`scale`、`lower_is_better`
- **为什么不能自己写**：`ref` 是官方 GPU 上的实测分数，是「劣化 ≤2%」判定的分母。自造等于伪造基线；「伪造日志」在反作弊条款里直接取消资格。
- **影响**：README §6 无法完成，`verify_result.json` 无法产出，**精度 Gate 无法通过**。
- **动作**：官方群索取，说明模型 ID `moonshotai/Kimi-Linear-48B-A3B-Instruct`、卡型 `BW1000`、`TP=4`。

### ② `data/` 评测子集 — 🔴 阻塞验收，只能官方给

- **要求出处**：同上，`data/` 为「模型官方下发的评测子集」
- **本件需要**：`data/gsm8k.jsonl`（`{"question","answer"}`）、`data/mmlu.jsonl`（`{"question","choices":[4],"answer"}`），题数须与 ① 中 `n` 完全一致（`verify.py` 会校验题数，防换子集）
- **能否自己造**：GSM8K / MMLU 是公开数据集，可以下到同分布样本，但**题数、抽样种子必须与官方一致**，否则分数不可比。
- **动作**：与 ① 一起索取。

### ③ T02–T04 分题材料 / 生成类与语音类性能测法 — 🟡 影响换赛道

- **要求出处**：《赛事详情》§五「### 通用材料（全赛道）」与「### 分题材料」两节**在原始文档中为空白**（只有标题，无任何表格或链接）；§七 同样的三节也为空白。
- **影响**：本件属 T01，暂不受影响；若要开 T02/T03/T04 的新件，缺评测器与测法。

## 已完成 / 可自行完成的 12 项

| # | 交付物 | 状态 | 说明 |
| --- | --- | --- | --- |
| 1 | `README.md` 8 章 | 🟢 7/8 章完成 | 第 6 章待 ① ② |
| 2 | `env.sh` | 🟢 | 全部为实测值 |
| 3 | `env_check.sh` → `env.log` | 🟢 | 在容器内采集 |
| 4 | `requirements-extra.txt` | 🟢 | 本件无额外依赖，仅说明性注释 |
| 5 | `patches/` | 🟢 | 本件无代码改动，留空 |
| 6 | `run.sh` → `server.log` | 🟢 | 已实跑通 |
| 7 | `smoke.sh` → `smoke.log` | 🟢 | health / models / 单轮 / 多轮 / 流式 |
| 8 | `verify.py` | 🟢 脚本就绪 | 判分逻辑自检通过；等 ① ② |
| 9 | `bench.sh` → `bench/` | 🟢 | 6 档 ×2 |
| 10 | `bench/` 产物 | 🟢 | round1/round2 jsonl + 各档日志 + summary.md |
| 11 | 服务可复现 | 🟢 | `container/submit.slurm` 一键起容器跑全流程 |
| 12 | `platform/` 证据 | 🟡 需人工截图 | Notebook/作业页、`hy-smi` 输出、服务映射页 |

## 另外两个非材料类阻塞（需要账号权限）

| 项 | 阻塞点 | 解法 |
| --- | --- | --- |
| 镜像推到 SCNet 镜像库 | `image.ac.com:5000`（Harbor）**匿名只能 pull，push 需要凭据** | 个人中心/镜像库页面取账号，`podman login` 后 `podman push` 到自有命名空间 |
| 代码推到光合托管 | `developer.sourcefind.cn/codes/modelzoo` 需写权限 | 确认账号与项目空间 |

## 附：这不是我们一家的问题（2026-10-30 实测）

`https://developer.sourcefind.cn/codes/explore/projects` 匿名可读，共 93 个公开提交仓库。抽查已提交的同题仓库：

| 仓库 | `verify.py` | `bench.sh` | `baselines/` |
| --- | --- | --- | --- |
| `linshp/mimo-sglang-bw1000`（BW1000 + SGLang + MiMo） | ❌ | ❌ | ❌ |
| `luql/mimo-sglang-bw1000`、`luql/mimo-vllm-bw1000`（本账号另两件） | ✅ | ✅ | ❌ |

其 README §6 原文：

> 官方 T01 赛道暂未针对该蒸馏模型下发固定的 GPU 基线文件（baselines/）及评测脚本（verify.py）。
> ……使用 GSM8K 和 MMLU 官方基准题型对 SGLang 接口进行了**自测**……待官方下发基线后，将随时按标准补充。

结论：**官方材料尚未下发给任何参赛者**，本仓库 §6 的阻塞是全赛道共性，不是个别遗漏。
