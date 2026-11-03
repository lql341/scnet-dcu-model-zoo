# 材料状态

## ✅ 已到（官方 T03 材料，2026-10-30 交付）

| 材料 | 位置 | 校验 |
| --- | --- | --- |
| 条件清单 60 条（20 首帧 × 3 seed） | `conditions/wan_i2v.jsonl` | 与官方 `formal/conditions/wan_i2v.jsonl` 同源 |
| 首帧素材 20 张（vbench_i2v） | 集群 `~/work/official-t34/data/vbench_i2v/`（不入库，99 MB 原包） | 入库时已按 `file-manifest.json` 校验 sha256，零缺失 |
| 反馈表模板 | `official/t03-t04/反馈表.csv` | — |

## ⚠️ 官方 pin 不可满足（以实测为准）

| # | 官方要求 | 实测情况 |
| --- | --- | --- |
| ① | `flash_attn`（INSTALL.md 措辞为"必须"） | DCU 无可用构建；`wan/modules/attention.py` 自动回落 torch SDPA |
| ② | `transformers>=4.49` | 镜像是 5.3.0。UMT5-XXL 能否加载以探针为准；退路是隔离装 4.57.6（`fa310` 那套已验证） |
| ③ | `diffusers>=0.31` | 隔离装 0.41.0（复用 Qwen-Image 归档轮子） |
| ④ | `numpy<2` | 镜像是 numpy 2.x。Wan2.1 的 numpy 依赖很浅（多为 imageio 调用），以探针为准 |

## ⬜ 未获官方下发

- `baselines/*.json`（T01/T02 的官方精度基线）——T03 本就不提供，官方明示不评分
- VBench 等自动评测器 —— 官方 T03 未提供，故本条目只交**原始产物 + 规格统计**，不自造指标

## 体积提示

60 条 × 81 帧 480p ≈ 每条 3–10 MB，合计约 300–600 MB，**不入库**（单条目上限 2 MB）。
入库物：`反馈表.csv`、`eval_result.json`、`stats.json`（帧数/fps/时长/体积，逐条）、`环境说明.txt`。
