# 材料状态

## ✅ 已到（官方 T03 材料）

| 材料 | 位置 | 校验 |
| --- | --- | --- |
| 条件清单 60 条（中英各 10 歌词 × 3 seed） | `conditions/music3.jsonl` | 与官方 `formal/conditions/music3.jsonl` 同源（sha256 已在 harness 的 evaluate 输出里记录） |
| 反馈表模板 | `official/t03-t04/反馈表.csv` | — |
| 素材 | 无需外部素材：歌词与描述都内联在条件清单里 | — |

## ⚠️ 官方未提供

- `baselines/*.json`：T01/T02 的精度基线（T03 本就不提供）
- 音乐质量评测器（FAD / CLAP 等）：官方 T03 明确不评分，故本条目只交原始产物 + 规格统计

## ⚠️ 依赖链风险（已实测通过，但仍是单点）

| # | 风险 | 现状 |
| --- | --- | --- |
| ① | 官方无独立推理仓，必须有 SGLang-Omni 服务端 | 用 Harbor 上 BW 专用镜像 + 镜像作者 harness（已归档到 `/public/home/luql/archive/zoo-dcu-2026-11/assets/`） |
| ② | 上游说需 2 张 CUDA 卡 | BW 单卡实测可用（TP=1，峰值 33.7 GB） |
| ③ | 镜像可能被上游删除 | 待归档镜像 tar（12.27 GB）；届时本条目不可复现 |

## 体积提示

60 条 × ~3.84 MB ≈ 230 MB，**不入库**（单条目上限 2 MB）。
入库物：`反馈表.csv`、`eval_result.json`、`环境说明.txt`、1 条样例 wav。
