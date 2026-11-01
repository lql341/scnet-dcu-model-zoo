# 作废数据存档 —— 首次压测（context-length=17408）

**这些数字不可用于报告**，保留作为「踩坑与规避」的证据。

## 发生了什么

`--context-length 17408` 恰好等于 `16384（输入）+ 1024（输出）`，但压测客户端的
`--random-input-len 16384` **不包含 chat template 的包装开销**，服务端实际收到
16740–18784 tokens，全部返回 HTTP 400：

```
Error: Requested token count exceeds the model's maximum context length of 17408 tokens.
       You requested a total of 18383 tokens: 17359 tokens from the input messages
       and 1024 tokens for the completion.
```

而压测客户端把 400 响应当作「请求完成」计入统计，于是产出了这些**物理上不可能**的数字：

```
16384 | 1024 | 1 | 8 | 0.84s | output_throughput 9791.30 tok/s | mean_ttft_ms 0.00
```

（0.84 秒内生成 8×1024 tokens 在 1.5 TFLOPS 的卡上不可能成立；TTFT 恒为 0 是请求立即被拒的特征。）

## 修正

1. `env.sh` 的 `MAX_LEN` 改为 `20480`
2. `bench.sh` 增加两道防护：
   - **前置校验**：压测前用长 prompt 发一个真实请求，非 200 直接退出
   - **有效性校验**：压测后检查 `mean_ttft_ms != 0` 且吞吐不超物理上限，否则判 INVALID 并拒绝生成 summary

## 存档文件

- `summary.BAD-ctx17408.md` —— 当时 `bench/summary.md` 的原文转录（数字作废，仅作记录）

原始 `round*.jsonl` 与分档日志已被重跑作业开头的 `rm -f bench/round*.jsonl` 清除，无法提供。
