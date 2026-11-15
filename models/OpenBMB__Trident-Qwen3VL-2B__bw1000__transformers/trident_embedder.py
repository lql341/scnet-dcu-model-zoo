#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trident_embedder.py -- single-file, clone-free inference for Trident-Qwen3VL.

Install (no repository clone needed):

    pip install torch "transformers>=4.57,<5" qwen-vl-utils==0.0.14 pillow numpy huggingface_hub
    # then either `pip install git+https://github.com/OpenBMB/Trident`, or just
    # download this one file next to your script.

Usage (mirrors the Qwen3VLEmbedder interface):

    from trident_embedder import TridentEmbedder

    model = TridentEmbedder("OpenBMB/Trident-Qwen3VL-2B")       # hub id or local dir

    queries = [{"text": "How many stores did Saint Laurent operate in Western Europe in 2020?"}]
    documents = [
        {"text": "A bar chart showing the number of Saint Laurent stores by region and year."},
        {"image": "https://raw.githubusercontent.com/OpenBMB/Trident/main/figs/example_chart.png"},
        {"text": "Number of stores by region.",
         "image": "https://raw.githubusercontent.com/OpenBMB/Trident/main/figs/example_chart.png"},
    ]

    q = model.process(queries, role="query")      # [1, dim] float32, L2-normalised
    d = model.process(documents, role="doc")      # [3, dim]
    print(q @ d.T)

Each input is a dict with any of:
    "text":        str
    "image":       one path / URL / data: URI / PIL.Image
    "images":      list of the above (use either "image" or "images")
    "instruction": optional override of the default instruction
    "role":        optional "query" / "doc"; overrides the `role` argument for this item
A text-only, image-only and text+image input are all valid and live in the same space.

Why `role` matters: Trident was trained with one instruction for queries and another
for documents, so queries must be encoded with role="query" and candidates with
role="doc". Mixing them up silently degrades retrieval.

This file mirrors src/mmemb/models/qwen3_vl_embedding.py and src/embedding_backends/
trident_qwen3vl.py of the repository (same prompt template, last-token pooling,
qwen_vl_utils resizing, pixel limits read from the checkpoint's mmemb_model.json).
Heavy imports are deferred to first use so the helpers can be imported cheaply.
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = ["TridentEmbedder"]

DEFAULT_MODEL = "OpenBMB/Trident-Qwen3VL-2B"

# Same defaults as configs/trident_qwen3vl.yaml; the checkpoint's mmemb_model.json wins if present.
DEFAULT_INSTRUCTIONS = {
    "query": "Represent the query for retrieving relevant content.",
    "doc": "Represent the candidate content for retrieval.",
}
IMAGE_BASE_FACTOR = 16          # qwen_vl_utils image_patch_size for Qwen3-VL
DEFAULT_MIN_PIXELS = 256 * 32 * 32
DEFAULT_MAX_PIXELS = 1280 * 32 * 32
DEFAULT_MAX_LENGTH = 8192
DEFAULT_MAX_TEXT_TOKENS = 4096

# Only these processor outputs are fed to the backbone (others, e.g. video kwargs, are dropped).
_MODEL_INPUT_KEYS = {
    "input_ids", "attention_mask", "position_ids", "inputs_embeds",
    "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
    "cache_position", "mm_token_type_ids",
}


# ---------------------------------------------------------------------- compat patch
def _ensure_special_tokens_compat() -> None:
    """Some tokenizer_config.json files store `extra_special_tokens` as a list (transformers v5
    format); transformers 4.x then fails with "'list' object has no attribute 'keys'".
    Convert the list to a {token: token} dict before the original method runs (idempotent)."""
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return
    original = getattr(PreTrainedTokenizerBase, "_set_model_specific_special_tokens", None)
    if original is None or getattr(original, "_trident_patch", False):
        return

    def patched(self, special_tokens, *args, **kwargs):
        if isinstance(special_tokens, list):
            special_tokens = {token: token for token in special_tokens}
        return original(self, special_tokens, *args, **kwargs)

    patched._trident_patch = True
    PreTrainedTokenizerBase._set_model_specific_special_tokens = patched


# ---------------------------------------------------------------------- pure helpers
def normalize_instruction(instruction: Optional[str]) -> Optional[str]:
    """Official behaviour: strip, and append '.' if the instruction does not end with punctuation."""
    if not instruction:
        return None
    instruction = instruction.strip()
    if not instruction:
        return None
    if not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def as_uri(path: Any) -> Any:
    """qwen_vl_utils expects `file://` for local files; URLs, data URIs and PIL images pass through."""
    if not isinstance(path, str):
        return path
    if path.startswith(("http://", "https://", "oss://", "data:", "file://")):
        return path
    return "file://" + os.path.abspath(os.path.expanduser(path))


def _images_of(item: Dict[str, Any]) -> List[Any]:
    images: List[Any] = []
    for key in ("image", "images"):
        value = item.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            images.extend(value)
        else:
            images.append(value)
    return images


def build_conversation(
    item: Dict[str, Any],
    instruction: str,
    min_pixels: int,
    max_pixels: int,
    truncate_text=None,
) -> List[Dict[str, Any]]:
    """Instruction goes in the system turn; images then text go in the user turn."""
    content: List[Dict[str, Any]] = []
    for image in _images_of(item):
        content.append({
            "type": "image",
            "image": as_uri(image),
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        })
    text = item.get("text")
    if text and truncate_text is not None:
        text = truncate_text(text)
    if text:
        content.append({"type": "text", "text": text})
    if not content:  # pooling needs at least one position
        content.append({"type": "text", "text": "NULL"})
    return [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": content},
    ]


def _load_json_from(name_or_path: str, filename: str) -> Optional[dict]:
    """Read a small JSON file from a local dir or the Hugging Face Hub; None if unavailable."""
    path = os.path.join(name_or_path, filename)
    if not os.path.isdir(name_or_path):
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(name_or_path, filename)
        except Exception:  # noqa: BLE001 - offline / file missing: fall back to defaults
            return None
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



# ---------------------------------------------------------------------- backbone
def _embedding_backbone_cls():
    """Qwen3VLForEmbedding: a Qwen3VLModel under the `model.` prefix (no lm_head).

    Defined here so loading never depends on the checkpoint shipping `auto_map` /
    modeling code. Weight keys (`model.visual...`, `model.language_model...`) are identical
    to checkpoints written by this repository's save_pretrained.
    """
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
    )

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping = {}
        accepts_loss_kwargs = False
        config_class = Qwen3VLConfig

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def get_input_embeddings(self):
            return self.model.get_input_embeddings()

        def set_input_embeddings(self, value):
            self.model.set_input_embeddings(value)

        def forward(self, **kwargs):
            return self.model(**kwargs)   # .last_hidden_state is all we need

    return Qwen3VLForEmbedding

# ---------------------------------------------------------------------- main class
class TridentEmbedder:
    """Clone-free wrapper: text / image / text+image  ->  L2-normalised embeddings."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        torch_dtype: Union[str, Any] = "bfloat16",
        attn_implementation: str = "sdpa",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_text_tokens: Optional[int] = None,
        dim: Optional[int] = None,
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_name_or_path: hub id (e.g. "OpenBMB/Trident-Qwen3VL-2B") or a local directory.
            device: "cuda:0", "cpu", ... Defaults to cuda if available, else cpu.
            torch_dtype: "bfloat16" / "float16" / "float32" or a torch.dtype.
            attn_implementation: "sdpa" (default) or "flash_attention_2" if flash-attn is installed.
            min_pixels / max_pixels: per-image pixel budget. Lower max_pixels first if you run out
                of memory. Defaults come from the checkpoint's mmemb_model.json.
            max_text_tokens: text is truncated to this many tokens when images are present.
            dim: optional Matryoshka truncation (first `dim` dims, re-normalised).
            batch_size: records per forward pass inside process().
        """
        import torch

        _ensure_special_tokens_compat()
        from transformers import AutoProcessor

        try:
            from qwen_vl_utils.vision_process import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "qwen-vl-utils is required for pixel-identical image preprocessing: "
                "`pip install qwen-vl-utils==0.0.14`"
            ) from exc
        self._process_vision_info = process_vision_info

        self.path = model_name_or_path
        self.dim = dim
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if isinstance(torch_dtype, str):
            torch_dtype = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }[torch_dtype.lower()]

        # Settings saved with the checkpoint (pixel limits, instructions, truncation).
        cfg = _load_json_from(model_name_or_path, "mmemb_model.json") or {}
        image_cfg = cfg.get("image") or {}
        prompt_cfg = cfg.get("prompt") or {}
        self.min_pixels = int(min_pixels or image_cfg.get("min_pixels") or DEFAULT_MIN_PIXELS)
        self.max_pixels = int(max_pixels or image_cfg.get("max_pixels") or DEFAULT_MAX_PIXELS)
        self.max_length = int(cfg.get("max_length") or DEFAULT_MAX_LENGTH)
        self.max_text_tokens = int(
            max_text_tokens or cfg.get("max_text_tokens") or DEFAULT_MAX_TEXT_TOKENS
        )
        self.instructions = {
            "query": prompt_cfg.get("query_instruction") or DEFAULT_INSTRUCTIONS["query"],
            "doc": prompt_cfg.get("doc_instruction") or DEFAULT_INSTRUCTIONS["doc"],
        }

        self.processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="right")
        if not hasattr(self.processor, "image_processor"):
            raise RuntimeError("Loaded a tokenizer-only processor; images would be silently dropped.")
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.model, info = _embedding_backbone_cls().from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        missing = info.get("missing_keys") or []
        if missing:
            raise RuntimeError(
                f"{len(missing)} weights were not found in {model_name_or_path!r} "
                f"(e.g. {missing[:3]}). The model would be randomly initialised, so embeddings "
                "would be meaningless. Check that the directory holds the full merged weights "
                "(model.safetensors with `model.*` keys), not a LoRA adapter or a different layout."
            )
        self.model.to(self.device).eval()

    # -------------------------------------------------------------- public API
    def process(
        self,
        inputs: Sequence[Dict[str, Any]],
        role: Optional[str] = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Embed a list of inputs -> float32 array [N, dim], L2-normalised, in input order.

        `role` is "query" or "doc" and applies to every item that has no own "role" key.
        """
        inputs = list(inputs)
        for item in inputs:
            r = item.get("role", role)
            if r not in self.instructions:
                raise ValueError(
                    "Specify role='query' or role='doc' (or a per-item 'role' key); "
                    f"got {r!r}. Queries and documents use different instructions."
                )
        if not inputs:
            return np.zeros((0, 0), dtype=np.float32)

        starts = range(0, len(inputs), self.batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                starts = tqdm(starts, desc="encode")
            except ImportError:
                pass

        chunks = [self._encode_batch(inputs[i:i + self.batch_size], role) for i in starts]
        return np.concatenate(chunks, axis=0)

    def embed_queries(self, inputs: Sequence[Dict[str, Any]], **kw) -> np.ndarray:
        return self.process(inputs, role="query", **kw)

    def embed_documents(self, inputs: Sequence[Dict[str, Any]], **kw) -> np.ndarray:
        return self.process(inputs, role="doc", **kw)

    @staticmethod
    def similarity(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
        """Cosine similarity [Nq, Nd] (embeddings are already L2-normalised)."""
        return np.asarray(queries) @ np.asarray(documents).T

    # -------------------------------------------------------------- internals
    def _truncate_text(self, text: str) -> str:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) <= self.max_text_tokens:
            return text
        return self.tokenizer.decode(ids[: self.max_text_tokens], skip_special_tokens=True)

    def _instruction_for(self, item: Dict[str, Any], role: str) -> str:
        return normalize_instruction(item.get("instruction")) or normalize_instruction(
            self.instructions[role]
        )

    def _build_inputs(self, items: List[Dict[str, Any]], role: Optional[str]) -> Dict[str, Any]:
        conversations = []
        for item in items:
            r = item.get("role", role)
            conversations.append(build_conversation(
                item,
                self._instruction_for(item, r),
                self.min_pixels,
                self.max_pixels,
                truncate_text=self._truncate_text,
            ))

        texts = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        if isinstance(texts, str):
            texts = [texts]

        try:
            images, _videos, _kwargs = self._process_vision_info(
                conversations,
                image_patch_size=IMAGE_BASE_FACTOR,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except TypeError:  # older qwen_vl_utils
            images, _videos = self._process_vision_info(conversations)

        kwargs: Dict[str, Any] = {
            "text": texts, "padding": True, "return_tensors": "pt",
            "do_resize": False,  # already smart-resized by qwen_vl_utils
        }
        if images:
            kwargs["images"] = images
        else:
            # Never truncate when images are present (would cut <|image_pad|> tokens).
            kwargs["truncation"] = True
            kwargs["max_length"] = self.max_length
        return dict(self.processor(**kwargs))

    def _encode_batch(self, items: List[Dict[str, Any]], role: Optional[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        batch = self._build_inputs(items, role)
        param_dtype = next(self.model.parameters()).dtype
        model_inputs = {}
        for key, value in batch.items():
            if key not in _MODEL_INPUT_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                value = value.to(self.device)
                if value.is_floating_point():
                    value = value.to(param_dtype)
            model_inputs[key] = value

        with torch.inference_mode():
            hidden = self.model(**model_inputs, use_cache=False).last_hidden_state
            mask = model_inputs["attention_mask"]
            # last valid token of each sequence (correct for left and right padding)
            idx = mask.long().cumsum(dim=-1).argmax(dim=-1)
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), idx].float()
            pooled = F.normalize(pooled, p=2, dim=-1)
            if self.dim and self.dim < pooled.size(-1):
                pooled = F.normalize(pooled[:, : self.dim], p=2, dim=-1)

        out = pooled.cpu().numpy().astype(np.float32)
        if out.shape[0] != len(items):
            raise ValueError(f"Model returned {out.shape[0]} embeddings for {len(items)} inputs.")
        return out


# ---------------------------------------------------------------------- zero-setup demo
def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Trident-Qwen3VL quick test")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    url = "https://raw.githubusercontent.com/OpenBMB/Trident/main/figs/example_chart.png"
    model = TridentEmbedder(args.model, device=args.device)

    queries = [{"text": "How many stores did Saint Laurent operate in Western Europe in 2020?"}]
    documents = [
        {"text": "A bar chart showing the number of Saint Laurent stores by region and year."},
        {"image": url},
        {"text": "Number of stores by region.", "image": url},
    ]
    names = ["text-only", "image-only", "text+image"]

    scores = model.similarity(model.process(queries, role="query"),
                              model.process(documents, role="doc"))[0]
    for name, score in zip(names, scores):
        print(f"{name:<11} {score:.4f}")
    print("ranking:", [names[i] for i in scores.argsort()[::-1]])


if __name__ == "__main__":
    _demo()
