# 16K/c64「每轮重启服务」原始数据（作业 1124366）

本目录是 `bench/round1.jsonl`、`bench/round2.jsonl` 之外的**补充数据**，
不是替代：那两份仍是本条目 6 档基准的正式来源，只是**不含 16K/c64**
（历史上两轮都崩，见 `../summary.md`「缺失档位」）。

## 为什么单独放

16K/c64 在本条目上受**服务进程累计状态**影响：同一进程连跑两轮，第二轮必崩
（5 个配置 5 个崩溃点，与并发数、cudagraph、批大小均无关）。
唯一稳定出数的方式是**每轮压测前重启服务进程**——本目录就是按这个口径跑出来的。

若把四轮数据直接塞进 `round1.jsonl`/`round2.jsonl`，会破坏那两份文件
"同一进程连跑两轮"的语义，也会让 `bench/summary.md` 与 `lint` 的逐字比对失真。
因此按 `docs/entry-spec.md` §6「数据留痕」的规则单开子目录。

## 数据

每份都是**单行**（一轮只跑一个档位），口径：`16384×1024`、`max_concurrency=64`、
`num_prompts=128`，与 `bench.sh` 其他档位完全一致。

| 文件 | Output throughput | Mean TTFT | Mean TPOT | Duration | 成功请求 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bench-S1-1124366.jsonl` | 235.14 tok/s | 47,489 ms | 218.3 ms | 557.42 s | 128/128 |
| `bench-S2-1124366.jsonl` | 283.53 tok/s | 31,498 ms | 187.4 ms | 462.29 s | 128/128 |
| `bench-S3-1124366.jsonl` | 284.62 tok/s | 30,841 ms | 187.2 ms | 460.52 s | 128/128 |
| `bench-S4-1124366.jsonl` | **286.84 tok/s** | **30,608 ms** | **185.7 ms** | 456.95 s | 128/128 |

四轮服务端均 **0 处 VMFault**。

## 引用口径

**取 R2–R4 的 283.53–286.84 tok/s（极差 1.2%）作稳态值。**
**不要把 R1 的 235.14 当稳态**——那是服务刚加载完模型、页缓存未暖的首轮，
比 R2 慢 17%，且 R1 的 TTFT 47.5 s 也明显偏高。

## 复现

```bash
./tools/sync_remote.sh                     # 必须先同步，否则远程 bench.sh 是旧版
sbatch sweeps/kimi-16k-020-restart.slurm   # NROUND=4，默认
```

详细分析见 `sweeps/results-kimi-16k-c64-2026-11-15.md` G 节与 `docs/gotchas.md` #51。
