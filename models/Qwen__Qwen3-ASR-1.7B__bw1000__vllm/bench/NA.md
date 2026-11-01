# 本条目未做 6 档基准

**原因**：赛题 §三 规定「文本 / 多模态理解类用 `bench.sh`，**生成类与语音类的测法随分题材料下发**」。
语音类的性能测法尚未下发，因此不适用通用的 4K/16K × 并发 1/8/64 六档负载
（语音请求是「音频时长 → 转写文本」，输入长度不以 token 计量）。

**已实测的性能相关数据**：

| 指标 | 值 |
| --- | --- |
| 冷启动 | 150–200 s |
| 显存占用 | `--gpu-memory-utilization 0.85`（模型仅 4.7 GB） |
| 单条 4 s 音频延迟 | 见 `logs/smoke.log`（`usage.duration`） |

**拿到官方测法后**：补 `bench.sh`、`bench/summary.md` 与 `round1/round2.jsonl`，
并把 `meta.json` 的 `bench` 从 `no-material` 改为 `yes`、清空 `bench_note`。
