# Trident-Qwen3VL-2B on BW1000 + SGLang 0.5.10 —— ❌ 框架不支持

> **结论：失败。** SGLang 0.5.10rc0 的 `qwen3_vl.py` 只注册了 `Qwen3VLForConditionalGeneration`，
> 没有 `Qwen3VLForEmbedding`。保留本条目作为证据。

## 1. 简介

`OpenBMB/Trident-Qwen3VL-2B` 是面壁智能的混合模态检索嵌入模型：把文本、图片、图文融合输入映射到同一向量空间，
使一条文本 query 能在同一列表里对纯文本 / 纯图片 / 图文文档统一排序。基于 Qwen3-VL-2B-Instruct 微调，LoRA 已合并。

`config.json` 实测：`architectures=['Qwen3VLForEmbedding']`、`model_type=qwen3_vl`，
BF16 单文件约 4.0 GB，MIT。

**它与同批其它 Qwen3-VL 模型的关键差别**：输出向量而非 token，架构名是 `ForEmbedding` 而不是 `ForConditionalGeneration`。

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1`（DTK 26.04 / torch 2.9.0 / SGLang 0.5.10rc0）

## 3. 适配优化方案

**无法适配** —— 失败在模型加载阶段的架构解析，参数无法绕过。

| 尝试 | 结果 |
| --- | --- |
| `--is-embedding --enable-multimodal --attention-backend triton` | ❌ 架构未注册 |

## 4. 服务启动脚本

`run.sh` 保留（若将来 SGLang 支持该架构可直接用），当前必然失败。

## 5. Smoke 测试

未执行，服务起不来。原始报错：

```
File ".../sglang/srt/model_loader/utils.py", line 54, in resolve_transformers_arch
    raise ValueError(
ValueError: Cannot find model module. 'Qwen3VLForEmbedding' is not a registered model
in the Transformers library (only relevant if the model is meant to be in Transformers)
and 'AutoModel' is not pre...
```

## 6. 精度测评

不适用。

## 7. 性能参考

不适用。

## 8. 注意事项

- **源码层面确认**：`sglang` 在 `v0.5.10rc0` / `v0.5.18` / `main` 三个版本的
  `python/sglang/srt/models/qwen3_vl.py` 里都只有 `EntryClass = Qwen3VLForConditionalGeneration`，
  没有 embedding 变体；也没有 `qwen3_vl_embedding.py` 文件（实测三个版本均 404）。
- **同一模型在 vLLM 0.20 上同样失败**，见 `../OpenBMB__Trident-Qwen3VL-2B__bw1000__vllm/`。
  两个主流框架都不支持，说明该架构目前只能走官方 `TridentQwen3VLBackend`（Transformers 直载）。
- **可行路径**：按官方 README 用 OpenBMB/Trident 的 `TridentQwen3VLBackend` 自建 FastAPI serving，
  并注意 query 与 doc 使用**不同的默认 instruction**（属训练格式，不能省）。
- 参考：`docs/gotchas.md` #17。
