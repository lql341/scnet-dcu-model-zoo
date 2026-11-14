# 仓库结构决策：为什么是单仓库

## 结论

**单仓库（monorepo）**，条目按 `models/<org>__<model>__<card>__<framework>/` 分目录。
参赛提交时用 `tools/export_contest.sh` 把单条目导出成独立仓库。

## 为什么不需要"每条目一个镜像"

这是决策的关键前提。**本项目从头到尾没有构建过任何镜像**：

| 事实 | 说明 |
| --- | --- |
| 19 个条目只用到 **5 个镜像** | SGLang / vLLM 0.20 / vLLM 0.21 / Music3 / HY-World，见各条目 `image.lock` |
| 镜像全部来自 Harbor 现成品 | `image.ac.com:5000/dcu/linshp/base/*`，公开项目、匿名可 pull |
| 使用方式是 `load` 或 `pull` | 本地缓存 tar（31 GB）或直接内网拉（4–6 min） |
| 没有任何条目需要 `Dockerfile` | 无一条目升级过包或打过 patch |

**如果每条目都需要自建镜像**（装包、编译 kernel、修残包），那才应该按条目拆库 —— 因为构建上下文和依赖会各自漂移。
当前不是这种情况，所以拆分只会带来 15 份 `bench.sh`、15 份 `preflight_lib.py` 的维护负担。

每条目附 `image.lock` 记录 **digest**，保证任意时间点复现出完全相同的环境；
`--tls-verify=false` 与 `--cert-dir` 的用法见 `docs/gotchas.md` #5。
若将来某条目必须自建镜像，在该条目目录放 `Dockerfile` 并在 README §3 说明即可，仍不需拆库。

## 脚本复用的量化事实

| 文件 | 各条目间的差异 |
| --- | --- |
| `bench.sh` | 完全相同（参数全部从 `env.sh` 读） |
| `preflight_lib.py` | 完全相同 |
| `verify.py` | 完全相同 |
| `env_check.sh` | 仅框架名列表不同 |
| `container/submit.slurm` | 卡数、工作目录、框架分支不同 |
| `env.sh` | **唯一实质差异**（模型路径、TP、MAX_LEN、EXTRA_ARGS） |
| `README.md` | 各自独立（技术报告） |

换句话说：**新增一个模型 = 新增一个 `env.sh` + 一份 README**，其余复用。
单次修复能覆盖全部条目 —— 例如 `/tokenize` 返回 500 那个 bug（`gotchas.md` #10），
一次改 `preflight_lib.py` 就让所有条目生效；如果是 15 个独立仓库，就要改 15 次。

## 体积控制策略

不靠感觉，靠脚本守门：`tools/check_quota.sh`

| 规则 | 阈值 |
| --- | --- |
| 单条目 | ≤ 2 MB |
| 全库 | ≤ 25 MB |
| 单文件（除 `bench/round*.jsonl`） | ≤ 1 MB |
| 大日志 | `server.log` 等 > 100 KB 一律 gzip（实测 552 KB → 40 KB，13:1） |
| 禁止入库 | 模型权重、镜像 tar、`__pycache__` |

当前**全库约 11 MB / 19 个条目**（含 bench 原始 jsonl 与日志），仍在 25 MB 阈值内。

保留明文的内容：`bench/round1.jsonl`、`bench/round2.jsonl`、`bench/summary.md`、`logs/env.log`、`logs/smoke.log`
（评审要能直接读）；压缩的只有冗长的 `server.log`。

## 什么时候该拆

只有出现下列情况之一时，才把某条目拆成独立仓库：

1. 该条目需要**自建镜像**（含 `Dockerfile`）且构建上下文 > 10 MB
2. 该条目需要**破坏性改公共脚本**（例如别处都要禁用 CUDA Graph，只有它要开）
3. 该条目的日志/产物**单条 > 10 MB**（例如超长跑批）
4. 参赛方明确要求仓库根目录必须是完整单件（届时用 `tools/export_contest.sh` 仍可从 monorepo 自动导出）

## 参赛提交与积累的关系

```
GitHub lql341/scnet-dcu-model-zoo          ← 权威积累（含 docs / tools / 横向对比）
        │  tools/export_contest.sh <条目目录>
        ▼
GitLab developer.sourcefind.cn/codes/luql/<件名>   ← 单件快照（评审点开即完整）
```

两边不手工同步：GitHub 是源头，导出脚本负责生成单件。
