# 材料状态

## ✅ 已到（官方 T03 材料）

| 材料 | 位置 |
| --- | --- |
| 条件清单 60 条（20 prompt × 3 seed） | `conditions/omniweaving.jsonl` |
| 反馈表模板 | `official/t03-t04/反馈表.csv` |

## ✅ 权重（含 4 份依赖，官方 download-checkpoint.md 的结构）

| 组件 | 来源 | 体积 |
| --- | --- | ---: |
| 主检查点（transformer/vae/upsampler/scheduler/text_encoder/ckpt） | `Tencent-Hunyuan/HY-OmniWeaving` | 55.3 G |
| `text_encoder/llm` | `Qwen/Qwen2.5-VL-7B-Instruct` | 16.6 G |
| `text_encoder/byt5-small` | `google/byt5-small` | 3.6 G |
| `text_encoder/Glyph-SDXL-v2` | `AI-ModelScope/Glyph-SDXL-v2` | 1.9 G |
| `vision_encoder/siglip` | `AI-ModelScope/FLUX.1-Redux-dev`（HF 原版是 gated） | 1.1 G |

## ⚠️ 官方未提供

- `baselines/*.json`（T01/T02 精度基线）——T03 不提供
- VBench 等自动评测器——官方 T03 明示不评分，故只交原始产物与规格统计

## ⚠️ 已知依赖冲突（已绕开）

| 官方要求 | 冲突 | 处置 |
| --- | --- | --- |
| `angelslim==0.2.1` | PyPI 已到 0.5.0，要求 transformers>=5.2；本条目用 4.57.1 | 不装（推理不需要） |
| `flash_attn` | 镜像自带版本 `unpad_input` 返回 4 项 | 运行时兼容补丁 `patches/0001-*` |

## ⏸ 未完成部分（非材料问题）

- 官方 60 条中 **7 条**因集群账号权限被回收而中断（2026-11-07），产物已归档 53 条到
  `luql:/public/home/luql/archive/zoo-dcu-2026-11/products/omniweaving/`
- 续跑命令：`sbatch --export=ALL,PROBE_ONLY=0 container/submit.slurm`

## 体积提示

60 条 × ~6 MB ≈ 350 MB，**不入库**；入库 `反馈表.csv`、`eval_result.json`、`stats.json`、环境说明。
