# Trident-Qwen3VL-2B on BW1000 + vLLM 0.20.0 —— ❌ 框架不支持

> **结论：失败。** vLLM 0.20.0 不认识 `Qwen3VLForEmbedding` 这个架构，服务无法启动。
> 保留本条目作为证据，不作为可用配置。

## 1. 简介

`OpenBMB/Trident-Qwen3VL-2B` 是面壁智能的混合模态检索嵌入模型（论文 *Chaos in the Text*，arXiv 2610.11816），
把文本、图片和图文融合输入映射到同一向量空间。基于 Qwen3-VL-2B-Instruct 在 Trident-Train 上微调，LoRA 已合并。

`config.json` 实测：`architectures=['Qwen3VLForEmbedding']`、`model_type=qwen3_vl`、`torch_dtype=bfloat16`，
BF16 单文件约 4.0 GB，MIT。

## 2. 验证环境

镜像：`image.ac.com:5000/dcu/linshp/base/vllm0.20-ubuntu22.04-dtk26.04-py3.10_qwen_asr1.7_bw:1.0.0`

```
torch 2.10.0 / HIP 6.3.26093 / vLLM 0.20.0 / transformers 5.8.1 / triton 3.5.0
Device: BW × 1，64.0 GB，gfx936
```

## 3. 适配优化方案

**无法适配** —— 失败发生在框架加载模型阶段，参数调整无法绕过。

尝试过的参数：

| 尝试 | 结果 |
| --- | --- |
| `vllm serve <model> --trust-remote-code` | ❌ 架构不被支持 |
| 追加 `--task embed` | ❌ `api_server.py: error: unrecognized arguments: --task`（vLLM 0.20 的 `api_server` 无此参数，嵌入模式由架构自动识别） |

## 4. 服务启动脚本

`run.sh` 保留（供将来 vLLM 支持该架构后直接可用），当前必然失败。

## 5. Smoke 测试

未执行 —— 服务起不来。失败证据见 §3 与 `logs/server.log`。

原始报错：

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, Model architectures ['Qwen3VLForEmbedding'] are not supported for now.
  Supported architectures: dict_keys(['AfmoeForCausalLM', 'ApertusForCausalLM', 'AquilaModel', ...])
```

## 6. 精度测评

不适用（服务无法启动）。

## 7. 性能参考

不适用。

## 8. 注意事项

- **这是架构级不兼容，不是配置问题**。`Qwen3VLForEmbedding` 需要框架显式注册该架构。
- **同一模型在 SGLang 上同样失败**，见 `../OpenBMB__Trident-Qwen3VL-2B__bw1000__sglang/`。
- **可行路径**：官方 README 要求用 OpenBMB/Trident 自带的 `TridentQwen3VLBackend`（Transformers 加载，
  `transformers >= 4.57`），即**自建 serving**而非用现成推理框架。若要完成该条目，需要写一个
  FastAPI 包装 + `TridentQwen3VLBackend`，并处理 query / doc 不同 instruction 的训练格式。
- 参考：`docs/gotchas.md` #17。
