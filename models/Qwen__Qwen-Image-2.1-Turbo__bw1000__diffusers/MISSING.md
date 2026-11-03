# 材料状态

## ✅ 已到

| 材料 | 位置 |
| --- | --- |
| 条件清单 180 条（60 输入 × 3 seed） | `official/t03-t04/conditions/qwen_image.jsonl` |
| 反馈表模板 | `official/t03-t04/反馈表.csv` |

## ✅ 尺寸口径已由条件清单确定

| 来源 | 尺寸 |
| --- | --- |
| 赛事详情 §三（性能测试） | 1024×1024 |
| 本次材料正文 | 「固定 2048×2048」 |
| **条件清单 `params.width/height`** | **2048 × 2048** ← 以机器可读的清单为准 |

## ✅ 2026-11-03 真实根因（此前被误判为「上游未来版本 API」）

| 症状 | 根因 | 修法 |
| --- | --- | --- |
| `ImportError: cannot import name 'resolve_revision'` | diffusers 0.41.0 要求 `huggingface-hub>=1.32.0,<2.0`；`--no-deps` 装的是 hub **0.36.2**，不报冲突 | 换 hub **1.33.0**（该版本 `resolve_revision` 与 `RevisionResolutionError` 都在） |
| `KeyError: 'MODEL_PATH'` | `container/submit.slurm` 没把 `MODEL_PATH` 传进容器 | 加 `-e MODEL_PATH="$MODEL_PATH"` |
| 探针按 40 步生成 | 未读条件清单的 `num_inference_steps` | 逐条读 `params`，见 `docs/gotchas.md` #20 |

`hub_shim.py` 已删除。另：diffusers **0.42.1 要求 Python ≥3.11**（容器 3.10）且 PyPI 不可解析，
故固定 0.41.0。

## 已完成

- ✅ 权重（32.5 GB）
- ✅ `diffusers 0.41.0 + huggingface_hub 1.33.0` 隔离安装于 `/opt/qi-libs`，导入通过
- ✅ 探针 6/6 成功：2048×2048，单图 **~20.7 s**（8 步 + KV cache + VAE 解码 + PNG）
- 🟡 官方 180 条全量跑在 `hx1hdnormal`（≈1 h/卡）

## 体积提示

产物 180 × ~6 MB ≈ 1 GB，**不入库**；只入库 `反馈表.csv`、`eval_result.json` 与若干缩略图。
