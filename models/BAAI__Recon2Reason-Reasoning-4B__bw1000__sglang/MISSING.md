# 官方尚未下发的材料

| # | 材料 | 影响 |
| --- | --- | --- |
| ① | T02 专属评测器（OCRBench / MMMU 子集脚本） | §6 无法按官方口径评测 |
| ② | `baselines/BAAI__Recon2Reason-Reasoning-4B.json` | 同上 |
| ③ | `data/` 评测子集（含图） | 同上 |

## 本模型的特殊性：可能不需要精度 Gate

赛题 §五 原文：

> （部分模型没有官方精度数据的，**可以不做**，如 `Recon2Reason-Reasoning-4B`、`RoboBrain2.5-8B-NV` 模型）

本模型在豁免名单内。但 T02 的**多模态链路 smoke**（单图 / 多图）仍须跑通，
本条目已完成链路验证（见 `logs/smoke.log`）。

## 行业现状

截至 2026-11-01，光合代码托管上 `recon2reason` 相关公开仓库 9 个，**没有一个**包含官方
`baselines/` 或 OCRBench/MMMU 基线分数。
