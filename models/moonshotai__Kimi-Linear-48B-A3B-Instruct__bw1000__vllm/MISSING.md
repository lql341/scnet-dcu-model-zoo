# 材料状态

## 🔄 官方已澄清（2026-11-15）

T01–T02 的验收标准**直接采用官方公开基线，不再单独下发** `baselines/<模型>.json` 与评测器。
因此本文件的阻塞性质从「等官方发」变成「从官方文档取用公开基线」：

- `verify.py` 读取路径不变（`baselines/<MODEL_ID 中 / 换成 __>.json`），
  取到官方基线数值后按原格式落盘即可跑通，**无需改代码**；
- 在取到之前，精度一栏仍保持 `[BLOCKED]`，**不填估算值**（红线 §6-1）。

## ⬜ 待取得（T01）

- 官方公开基线中的子集、max_tokens、思考开关、GPU 参考分
  → 取得后填入 `baselines/moonshotai__Kimi-Linear-48B-A3B-Instruct.json`

## ✅ 已有

- 权重 92 GB（ModelScope，`moonshotai/Kimi-Linear-48B-A3B-Instruct`）
- 6 档压测客户端（SGLang 镜像内 `sglang.benchmark.serving`）
- 配对条目：SGLang 版本的完整 6 档数据（`../moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang/bench/`）

## 体积提示

6 档日志与 jsonl 约 1 MB，入库；权重不入库。
