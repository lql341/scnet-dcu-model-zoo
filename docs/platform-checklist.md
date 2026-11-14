# `platform/` 截图清单

赛题模板 §2 最后一行要求提供「平台证据」，各条目的 `platform/` 目录因此为空。
本清单说明每个条目需要哪几张图，以及每张图该出现什么内容。

## 通用 4 张（所有条目都要）

| # | 截图 | 必须出现的内容 | 用途 |
| --- | --- | --- | --- |
| 1 | **Slurm 作业提交页** | `sbatch` 输出 / `squeue` 显示 job id、分区 `hx1hdnormal`、`gres/dcu:N`、状态 | 证明用了 SCNet 算力 |
| 2 | **`hy-smi` 输出** | 卡型号 `BW`、显存 `65520 MiB`、`PwrCap 1000W`、占用率 | 证明卡型与赛题一致 |
| 3 | **作业完成后 `sacct`** | `State=COMPLETED`、`Elapsed`、`AllocTRES` 含 `gres/dcu=N` | 证明实际跑了多久、用了什么 |
| 4 | **模型文件列表** | `ls -l <模型路径>` 显示分片数与总大小 | 证明权重来源与规模 |

## 条目专属

| 条目 | 额外需要的图 |
| --- | --- |
| Kimi-Linear（TP=4） | `hy-smi` 显示 **4 张卡同时 86%** 占用 |
| MiMo × SGLang / vLLM | `hy-smi` 单卡 **86%/94%** 占用（两份优化前后各一张，能看出差异更好） |
| Qwen3-ASR-1.7B | 300 条评测完成时的 `outputs/` 文件数（300 个 txt） |
| Qwen3-ForcedAligner | 30 条产出的 `outputs/` 计数（60 个 json+tsv） |
| Recon2Reason / RoboBrain | 多图请求时 `server.log` 里 `prompt_tokens` 从 67→115 的日志行 |
| Trident（blocked） | 报错截图（`ValueError: 'Qwen3VLForEmbedding' is not a registered model`） |

## 服务映射

本次全部条目均在 Slurm 作业内以容器 + `127.0.0.1:30000` 自测，**未对外暴露服务**。
按模板 §2 的表述，此项填「无」，并附一句说明：

> 服务映射：无。所有条目在 Slurm 作业内以容器方式运行，服务监听 `127.0.0.1:30000`，
> 由 `smoke.sh` / `bench.sh` 在同一作业内访问，未映射到外部。

## 采集命令（可直接执行）

```bash
# 1. 作业提交
sbatch --parsable container/submit.slurm | tee platform/01-submit.txt

# 2. 运行中
squeue -j <JOBID> -o "%.8i %.9P %.9T %.7M %R" | tee platform/02-squeue.txt
ssh <node> 'hy-smi' 2>/dev/null          # 或在作业脚本内 hy-smi | tee

# 3. 完成后
sacct -j <JOBID> --format=JobID,State,Elapsed,AllocTRES | tee platform/03-sacct.txt

# 4. 模型
ls -l <HOST_MODEL_PATH> | tee platform/04-model.txt
```

## 现状

| 条目 | platform/ 内容 |
| --- | --- |
| 全部 19 个条目 | ⬜ 空（仅 `platform/EVIDENCE.txt`，无截图） |

**这不是技术问题**：所有证据都能从已有日志里重建（`logs/server.log` 里有 `hy-smi` 输出，
`env.log` 中有卡型与显存）。文本版证据（`platform/EVIDENCE.txt`）已由现有日志生成，
但赛事要求为截图，仍需在网页或终端中实际截取一次。
