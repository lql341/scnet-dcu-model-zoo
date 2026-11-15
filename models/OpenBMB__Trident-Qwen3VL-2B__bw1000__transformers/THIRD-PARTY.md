# 第三方文件

## `trident_embedder.py`

- 来源：模型仓库 `OpenBMB/Trident-Qwen3VL-2B` 内的同名文件（HF revision 见 `image.lock` 同目录 `weights.lock`）
- 作者：OpenBMB
- 许可：**MIT**（模型仓库 `license: mit`）
- 作用：单文件、免 clone 的 Trident 推理封装，镜像官方仓库 `src/mmemb/models/qwen3_vl_embedding.py`
  与 `src/embedding_backends/trident_qwen3vl.py`（同一 prompt 模板、last-token pooling、
  `qwen_vl_utils` 尺寸策略、像素上限读自检查点的 `mmemb_model.json`）
- 未做任何修改（逐字节一致）；sha256 见 `weights.lock` 的 `trident_embedder.py` 字段

**为什么必须用它**：检查点的 `config.json` 声明 `architectures=["Qwen3VLForEmbedding"]`，
但**没有** `auto_map`，即权重本身不带建模代码；而 vLLM 0.20 / SGLang 0.5.10 都没有注册
`Qwen3VLForEmbedding`（见本仓库两个 blocked 条目）。这个文件在运行时用
`transformers.models.qwen3_vl.modeling_qwen3_vl` 现拼一个同名类，从而绕开注册表。
