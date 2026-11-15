#!/usr/bin/env python3
"""图文路径诊断：确认 pixel_values 的 reshape 不匹配发生在预处理还是模型代码。

背景：容器内 text_only 正常（dim=2048、已 L2 归一化），但任何图像输入都报
    RuntimeError: shape '[1, 1, 2, 3, 32, 2, 16, 32, 2, 16]' is invalid for input of size 6439776
需要区分两种情况：
  A) processor 产出的 pixel_values 与 image_grid_thw 自相矛盾（预处理/版本不配套）
  B) processor 一致，是 modeling 侧把 patch 展开算错（模型代码/transformers 版本）

用法：python3 diag_images.py <model_dir>
"""
from __future__ import annotations

import sys

import torch
from PIL import Image


def main() -> None:
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "/models/Trident"
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print("[diag] processor:", type(proc).__name__)

    for size in (320, 512, 1024):
        img = Image.new("RGB", (size, size), (200, 30, 30))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": "a red square"}]}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        out = proc(text=[text], images=[img], return_tensors="pt")
        pv = out.get("pixel_values")
        thw = out.get("image_grid_thw")
        print(f"[diag] size={size} pixel_values={tuple(pv.shape)} image_grid_thw={thw.tolist() if thw is not None else None}")
        if pv is not None and thw is not None:
            t, h, w = thw[0].tolist()
            # 期望元素数 = t * (h/merge) * (w/merge) * C * temporal * patch * patch
            cfg = proc.image_processor
            ps = getattr(cfg, "patch_size", 16)
            merge = getattr(cfg, "merge_size", 2)
            tmp = getattr(cfg, "temporal_patch_size", 2)
            expect = t * (h // merge) * (w // merge) * 3 * tmp * ps * ps
            print(f"[diag]   期望 {expect} 元素，实际 {pv.numel()} → {'一致' if expect == pv.numel() else '★不一致★'}")

    # 直接跑一次模型（绕过 HTTP 层），把真实异常打出来
    sys.path.insert(0, "/work/repo")
    from trident_embedder import TridentEmbedder
    emb = TridentEmbedder(model_dir, batch_size=1)
    for size in (512, 1024):
        img = Image.new("RGB", (size, size), (200, 30, 30))
        try:
            v = emb.process([{"text": "a red square", "image": img}], role="doc")
            print(f"[diag] embed size={size} OK shape={v.shape}")
        except Exception as e:
            print(f"[diag] embed size={size} FAIL {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
