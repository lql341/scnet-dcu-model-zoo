# OmniWeaving 多卡序列并行：4 卡仍未跑通（2026-11-12）

## 目标

单卡 24.6 min/条 × 60 条 = 24.6 h。官方 `generate.py` 用 `WORLD_SIZE` 做序列并行（SP），
理论 4 卡可到 ~7 min/条，是当前剩余优化里 ROI 最高的一项（最多 4×）。

## 已做的修正

| # | 问题 | 修法 | 结果 |
| --- | --- | --- | --- |
| 1 | 上游 `generate.py` **从头到尾没有 `init_process_group`**，多卡时 `build_mesh` 直接崩 | `patches/0002-init-process-group.patch`：`WORLD_SIZE>1` 时先 `dist.init_process_group('nccl')` | ✅ 4 个 rank 能起来 |
| 2 | Slurm 只给 1 张卡的可见性，其余 rank 越界 | inner.sh 多卡分支 `env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES -u ROCR_VISIBLE_DEVICES` | ✅ 不再报 device ordinal |
| 3 | 探针被"已有产物"跳过，等于没测 | `PROBE_FRESH=1` 输出到 `official-eval/probe-outs/`（不影响正式产物） | ✅ 探针真的跑 |

## 当前结果

```
[omni] PROBE_FRESH=1 → 输出到 official-eval/probe-outs
[omni] 多卡模式：NPROC=4，unset 可见性变量
[miss] 总 60，待跑 1
[miss] 1/1 FAIL vbench_00000_s42 rc=1 size=0
```

torchrun 侧报 `rank : 2 (local_rank: 2) exitcode: 1`，但**真正的 traceback 没抓到**：
`run_missing.py` 把子进程输出写进容器内的 `/tmp/gen-<id>.log`，容器销毁即丢。

## 下一步（下次只需 30 分钟）

1. `run_missing.py` 的失败日志改写到 `$OUTDIR/../logs/gen-<id>.log`（宿主持久化），
   或直接用 `TORCHELASTIC_ERROR_FILE` + `--redirects 1`（Wan 条目已这么做）
2. 再跑一次 `PROBE_ONLY=1 PROBE_FRESH=1`，读 rank 级 traceback
3. 常见嫌疑：`WORLD_SIZE` 与 `sp` 的 mesh 维度不匹配、或 vision/VAE 未按 rank 分片

## 结论（截至 2026-11-12）

- **多卡 SP 尚未跑通**，条目仍按单卡 60/60 交付（结果有效，只是慢）
- 三个前置问题（缺 init_process_group / 可见性 / 探针跳过）已修掉，剩下的是 rank 级报错定位
