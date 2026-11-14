# 权重管理策略（集群配额只有 400 GB）

## 原则

**权重不是资产，镜像和 wheel 才是。** 权重可从 ModelScope 重下（实测 40–50 MB/s），
因此「已出官方结果的模型」用完就删，把配额留给还没跑通的东西。

## 什么该删 / 什么该留

| 类别 | 处置 | 理由 |
| --- | --- | --- |
| 已产出官方结果的条目 | ✅ **可删**（结果已在 `official-eval/`） | 需要复跑时重下 10–20 min |
| 正在跑 / 下一步要跑 | ❌ 保留 | — |
| 只做 smoke、未出结果 | 🟡 保留（体积小）或删（体积大且无后续计划） | — |
| 曾被判定 blocked 且无新证据 | ✅ 可删 | 结论已入库，权重留着没用 |
| 上游镜像（tar.gz）与 wheel 集 | ❌ **永远保留**（在 `/public/home/luql/archive/zoo-dcu-2026-11/`） | 不可重下：上游随时删仓，且受 quota 影响 |

## 删除记录

| 日期 | 删除 | 大小 | 原因 | 重下命令 |
| --- | --- | ---: | --- | --- |
| 2026-11-03 | `Kimi-Linear-48B-A3B-Instruct` | 92 G | 给 Wan2.1（82 G）+ OmniWeaving（55 G）腾配额 | `python3 -c "from modelscope import snapshot_download as d; d('moonshotai/Kimi-Linear-48B-A3B-Instruct', local_dir='/public/home/scnethr2601a10/models/Kimi-Linear-48B-A3B-Instruct')"` |
| 2026-11-04 | `AuK`、`AuK-Flash` | 12.6 G | 两个条目均已 120/120 完成并入库 | `d('Tencent-Hunyuan/AuK', …)` / `d('Tencent-Hunyuan/AuK-Flash', …)` |
| 2026-11-04 | `Qwen2.5-Omni-3B` | 12 G | 仅作 AuK/AuK-Flash 的文本编码器，两者已完成 | `d('Qwen/Qwen2.5-Omni-3B', …)` |
| 2026-11-04 | `Trident-Qwen3VL-2B` | 4 G | 架构不被 SGLang/vLLM 支持，已如实入库为 blocked | `d('OpenBMB/Trident-Qwen3VL-2B', …)` |
| 2026-11-04 | `HY-World-2.0`（WorldMirror 子集 5 G） | 5 G | 需自编 ROCm 版 gsplat，暂缓 | `d('Tencent-Hunyuan/HY-World-2.0', allow_patterns=['HY-WorldMirror-2.0/*'])` |

**当前保留**：Qwen-Image 31 G（等做合批优化）、Music3 54 G（待跑）、OmniWeaving 52 G（待跑）、
Wan2.1 77 G（正在跑全量）、MiMo 18 G、RoboBrain 17 G、Recon2Reason 8.3 G、ASR 4.4 G、ForcedAligner 1.8 G。

## 删之前必须确认

1. 该条目的 `official-eval/`（或 `bench/`）已入库并推送双远端
2. `meta.json` 的 status 已更新到最终状态
3. 把删除记录追加到本文件（含重下命令）——否则下个会话会以为权重还在
