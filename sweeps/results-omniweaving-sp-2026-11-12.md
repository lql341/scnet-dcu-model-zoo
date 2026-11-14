# OmniWeaving 多卡序列并行：4 卡已跑通，2.23× 提速（2026-11-12 起，11-14 结论）

## 目标

单卡 **24.6 min/条**（1476 s），60 条需 24.6 h。官方 `generate.py` 用 `WORLD_SIZE` 做序列并行（SP），
理论 4 卡可到 ~7 min/条，是当时剩余优化项里 ROI 最高的一条（最多 4×）。

## 最终结果（2026-11-14 实测）

4 卡 SP，环境 `--gres=dcu:4 -c 64 --mem=240G`，容器内 `device_count()=4`：

| 条件 | 单条耗时 |
| --- | ---: |
| `vbench_00000_s42` | **698 s** |
| `vbench_00000_s2026` | 661 s |
| `vbench_00000_s1234` | 647 s |
| `vbench_00001_s42` | 640 s |
| 均值 | **661.5 s（11.0 min）** |
| 对比单卡 1476 s | **2.23×**（并行效率 56%） |

- 全量 60 条外推 **≈11.1 h**（60 × 661.5 s + 6 min 权重加载），比单卡 24.6 h 少 **13.5 h**。
  该外推未实跑全量：单卡 60/60 产物已验证有效，重跑只增耗时、不增证据。
- 产物规格与单卡一致：4 个 mp4 时长均 **5.06 s**、81 帧、832×480，`spec_violation_count=0`。
- 未达到理论 4×：SP 只切 DiT 主干，**编码器与 VAE 每 rank 各持一份**，且存在 rank 间同步开销；
  2.23× 与"主干占单条耗时约一半"的经验分布吻合。

## 三类阻塞与修法（按发现顺序）

| # | 现象 | 根因 | 修法 | 证据 |
| --- | --- | --- | --- | --- |
| 1 | 多卡时 `build_mesh` 直接崩 | 上游 `generate.py` **全程没有 `init_process_group`** | `patches/0002-init-process-group.patch`：`WORLD_SIZE>1` 时先 `dist.init_process_group('nccl')` | 4 个 rank 可启动 |
| 2 | `rank 2 exitcode 1`，抓不到 traceback | `run_missing.py` 把子进程日志写进容器 `/tmp`，容器销毁即丢 | 日志改写到 `$OUTDIR/../genlogs/`，并加 `--redirects 1` + `TORCHELASTIC_ERROR_FILE`，失败时打印全 rank 关键行 | `genlogs/gen-*.log` 持久化 |
| 3 | rank 1/2/3 报 `HIP error: invalid device ordinal` | 申请 `dcu:4` 但 `-c` 只有 16，Slurm 按 socket 绑卡时**把 GRES 降级到 1 张**（`AllocTRES` 才是准的） | 每卡 ≥8 CPU：`-c 64` 配 4 卡 | `sacct -o ReqTRES,AllocTRES` |
| 4 | rank 2 `exitcode -9`，无 traceback，宿主 `oom_kill events` | 4 个 rank 各持 VAE + 文本/视觉编码器并 offload，≈200 GB；`-c 32` 自动带出 `--mem=121G` | `-c 64 --mem=240G`（受 `DefMemPerCPU=3888M` 约束，mem ≤ c×3888M） | `slurmstepd: Detected 2 oom_kill events` |

第 3 条是本次最贵的一环：先后在"设备可见性"方向上试了 6 组配置
（unset 变量、容器内显式赋值、podman 级 `-e`、逐节点 `--device`、挂 `/sys/class/kfd` 与 `/sys/class/drm`、
`--group-add keep-groups`），容器内 `hy-smi -L` 始终只列 1 张卡——直到查 `AllocTRES` 才定位。
详见 `docs/gotchas.md` 第 30–32 条。

## 复现命令

```bash
cd models/Tencent-Hunyuan__HY-OmniWeaving__bw1000__hyvideo
sbatch container/submit-4card.slurm                                   # 正式产物
sbatch --export=ALL,PROBE_ONLY=4,OUTDIR_OVERRIDE=official-eval/sp4bench container/submit-4card.slurm   # 计时
```

`submit.slurm`（单卡）仍是默认路径：60/60 官方产物即由它产出，改动不影响既有结果。

## 结论

- **4 卡 SP 可用**，条目性能口径由「单卡 24.6 min/条」升级为「单卡 24.6 min/条 / 4 卡 11.0 min/条」。
- 多卡的门槛不在模型侧，而在**集群配额语义**（CPU:DCU 比例、`DefMemPerCPU`），这两条已写进踩坑清单，
  可直接复用到 Wan2.1 8 卡、Kimi 16K 等后续多卡实验。
