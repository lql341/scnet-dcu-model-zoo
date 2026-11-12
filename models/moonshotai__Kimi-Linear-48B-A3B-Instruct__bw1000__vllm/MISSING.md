# 材料状态

## ⬜ 官方未下发（T01）

- `baselines/Kimi-Linear-48B-A3B-Instruct__*.json`（官方子集、max_tokens、思考开关、GPU 参考分）
  → 精度一栏保持 `[BLOCKED]`，不填估算值

## ✅ 已有

- 权重 92 GB（ModelScope，`moonshotai/Kimi-Linear-48B-A3B-Instruct`）
- 6 档压测客户端（SGLang 镜像内 `sglang.benchmark.serving`）
- 配对条目：SGLang 版本的完整 6 档数据（`../moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/bench/`）

## 体积提示

6 档日志与 jsonl 约 1 MB，入库；权重不入库。
