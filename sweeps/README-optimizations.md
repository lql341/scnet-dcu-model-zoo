# 2026-11-05 优化实验清单（进行中）

| 脚本 | 目的 | 状态 |
| --- | --- | --- |
| `mimo-sglang-16k-opt.slurm` | MiMo×SGLang 16K：chunked-prefill 与 CUDA Graph **组合**（历史只单独测过） | ✅ 已出结论 |
| `mimo-confirm.slurm` | 用 base vs chunk4k 各 2 轮确认增益是否真实 | 🟡 反复被节点 /tmp 水位打断 |
| `kimi-16k-opt.slurm` | Kimi 16K/c64 回滚修复（cuda-graph-max-bs / mem 0.92 / chunk4k / 无 graph 对照） | 🟡 跑起来但 bench 无产物，待查 |
| `vllm-v0-compare.slurm` | vLLM `VLLM_USE_V1=0` 对比 | ✅ 结论：与 v1 完全一致（无效） |
| `music3-restore-verify.slurm` | Music3 归档功能性还原 | ✅ 通过（32 kHz/2ch/30.02 s） |
| `dcu-visible-probe.slurm` | 诊断 4 卡作业里 torch 实际可见几张卡 | 🟡 首个探针因 LD_LIBRARY_PATH 不全而失败，结论改用 scontrol 读 gres |

## 已确定的结论（MiMo×SGLang 16K）

一轮扫描（每档单轮，同节点同参数）：

| 变体 | 16K/c64 | 16K/c8 |
| --- | ---: | ---: |
| A base（graph 开，无 chunked） | 279.56 | 217.63 |
| **B chunked-prefill 4096** | **309.90（+10.9%）** | 223.36（+2.6%） |
| C chunked 8192 | 299.35 | 218.13 |
| D mrr 128 | 300.20 | 217.69 |

对照官方 bench 记录（299.81 / 308.72），单轮波动约 ±7%，**所以 +10.9% 必须做 2 轮确认才能写进条目**。
`mimo-confirm.slurm` 就是干这个的；反复被节点磁盘水位打断，等集群空闲再跑。

## 环境层面的新发现

- 本集群 `sbatch` 命令行的 `--gres` 覆盖无效，卡数只能写进文件头；
- `scontrol show job` 报告 `gres/dcu=4`，但 `sacct AllocTRES` 可能显示 `gres/dcu=1`（记账不准）；
- 同时 `CUDA_VISIBLE_DEVICES` 只给 `0`，因此**容器内 torch 默认只看到 1 张卡**
  → 多卡 FSDP/SP 必须在容器内 `env -u` 掉可见性变量（Wan 已验证），OmniWeaving 的 SP 仍待解决。
