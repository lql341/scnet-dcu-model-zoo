# 材料状态

## 🔄 官方已澄清（2026-11-15）

T01–T02 的验收标准**直接采用官方公开基线，不再单独下发**评测器与 `baselines/<模型>.json`。
阻塞性质从「等官方发」变成「从官方文档取用公开基线」，但**口径不可比**（见
`docs/t01-t02-baseline-investigation.md`）。

## ⬜ 待取得

- T02 评测器（OCRBench / MMMU）与 `data/` 子集 → 未取得前精度保持 `[BLOCKED]`
- 官方公开基线中的 GPU 参考分（且需先确认 shot 数与本仓库 0-shot greedy 是否可比）

## ✅ 已有

- 权重（ModelScope，16.34 GiB / 17 文件）—— 获取方式见下节
- 同底座先例：`BAAI__RoboBrain2.5-8B-NV__bw1000__vllm` 已实测 vLLM 0.21 单图/多图均 200
- 本条目**没有** `__sglang` 配对条目：同底座的 SGLang 侧已因 Triton VMFault 封存（gotchas #57/#58），
  不再重复投卡时验证同一失败

## 权重获取（与其它条目不同，勿照抄旧流程）

建条目时（2026-11-22）集群**已无外网**：登录节点与计算节点 DNS 均不可用，公网 IP 的 443 全部超时，
内网 pip 源也拒绝连接（证据见 `sweeps/results-cluster-egress-2026-11-22.md`）。
因此**不能**在集群上 `snapshot_download`——历史条目（Kimi 2026-11-05、DeepSeek-V4.1 2026-11-15）
当时还能直连 ModelScope，现在这条路径已失效。

现行做法：本机下载（实测 16 MB/s）→ OpenAPI 单文件上传（实测 10 MB/s），全程约 50 分钟：

```bash
# 本机：按清单逐文件 curl -C - 续传，校验字节数
# 上传：scnet.py --region 20091 upload <file> /public/home/luql/models/Qwen3-VL-8B-Instruct/
```
