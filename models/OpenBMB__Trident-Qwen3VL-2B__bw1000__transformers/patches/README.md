# 本条目没有补丁

Trident-Qwen3VL-2B 走的是官方 `trident_embedder.py`（MIT，未改动），
它用 `transformers.models.qwen3_vl.modeling_qwen3_vl` 在运行时现拼一个
`Qwen3VLForEmbedding` 类，因此不需要给框架打补丁。

`trident_embedder.py` 的确切来源与 sha256 见 `../THIRD-PARTY.md`。

（空目录保留 .gitkeep，因为条目规范要求存在 `patches/`。）
