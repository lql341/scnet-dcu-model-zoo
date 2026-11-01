# 作废：首次压测结果（context-length=17408）

> ⚠️ 本表数字**全部不可用于报告**。保留仅为记录「发现并修正问题」的过程。
> 原始 jsonl 与分档日志已被后续作业的 `rm -f bench/round*.jsonl` 清除，
> 此处为当时 `bench/summary.md` 的原文转录。

| random_input_len | random_output_len | max_concurrency | completed | duration | request_throughput | output_throughput | total_throughput | mean_ttft_ms | p99_ttft_ms | mean_tpot_ms | p99_tpot_ms | mean_e2e_latency_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1024 | 1 | 8 | 328.15 | 0.02 | 24.96 | 124.82 | 353.98 | 371.05 | 39.75 | 40.30 | 41017.47 |
| 4096 | 1024 | 8 | 16 | 86.94 | 0.18 | 188.45 | 942.25 | 2168.60 | 3114.67 | 40.37 | 42.37 | 43463.13 |
| 4096 | 1024 | 64 | 128 | 713.33 | 0.18 | 183.75 | 918.73 | 236188.03 | 315463.19 | 41.28 | 43.17 | 278416.99 |
| 16384 | 1024 | 1 | 8 | 0.84 | 9.56 | **9791.30** | 166452.12 | **0.00** | 0.00 | 0.10 | 0.12 | 103.05 |
| 16384 | 1024 | 8 | 16 | 1.70 | 9.43 | **9652.12** | 164086.10 | **0.00** | 0.00 | 0.67 | 0.91 | 682.26 |
| 16384 | 1024 | 64 | 128 | 13.26 | 9.66 | **9886.81** | 168075.80 | **0.00** | 0.00 | 4.83 | 6.41 | 4936.33 |

## 无效判据

- `16384` 三行的 `mean_ttft_ms` 恰好为 **0.00** —— 请求被立即拒绝的特征
- `16384` 并发 1 档 8 个请求（每请求 16384 输入 + 1024 输出）在 **0.84 s** 内完成，
  按 `output_throughput 9791 tok/s` 推算需要 >10× 于该卡物理峰值
- 同一节点上 `4096` 三行数据正常（24–188 tok/s，TTFT 正常），说明卡本身没问题

## 根因

服务端 `--context-length 17408` 小于客户端实际发出的 16740–18784 tokens
（`--random-input-len` 不计 chat template 包装开销）：

```
[http_server] Error: Requested token count exceeds the model's maximum context length of 17408 tokens.
              You requested a total of 18383 tokens: 17359 from input + 1024 for completion.
```

压测客户端把 HTTP 400 计为「完成」，因此 `completed` 仍是 8/16/128，掩盖了失败。

## 修正后

`MAX_LEN=20480` + `bench.sh` 前置 preflight + 结果有效性校验，重跑结果见 `../summary.md`。
