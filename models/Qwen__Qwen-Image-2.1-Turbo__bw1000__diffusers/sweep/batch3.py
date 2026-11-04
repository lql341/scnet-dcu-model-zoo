#!/usr/bin/env python3
"""Qwen-Image 合批优化实验：同一 prompt 的 3 个 seed 合成一次 pipeline 调用。

背景：官方条件 60 prompt × 3 seed = 180 张，逐张生成时每张都要重跑一遍文本编码 +
VAE 解码框架；而 3 个 seed 的 prompt/文本编码完全相同，可以合并成一次
`num_images_per_prompt=3` 的调用（`generator` 传 3 个 CPU Generator）。

产物与官方官方条件**完全一致**（同 prompt、同 seed、同尺寸/步数/guidance），
只是**执行方式**不同，用于衡量吞吐优化空间；结果写 `bench-batch/`，不覆盖官方产物。

用法：
  BATCH_SEEDS=3 PROBE_PROMPTS=6 python3 sweep/batch3.py     # 先跑 6 个 prompt（18 张）
  BATCH_SEEDS=3 python3 sweep/batch3.py                     # 全量 60 个 prompt（180 张）
"""
import json, os, sys, time
from collections import OrderedDict

COND = os.environ.get("COND_FILE", "conditions/qwen_image.jsonl")
OUT = os.environ.get("OUTDIR", "bench-batch/outputs")
PROBE = int(os.environ.get("PROBE_PROMPTS", "0") or 0)
MAX_BATCH = int(os.environ.get("MAX_BATCH", "3") or 3)   # 2048² 下单卡显存决定；实测 3 会 OOM
RESUME = os.environ.get("RESUME", "1") == "1"


def main():
    import torch
    from diffusers import QwenImage21Pipeline

    items = [json.loads(l) for l in open(COND, encoding="utf-8") if l.strip()]
    groups = OrderedDict()
    for it in items:
        groups.setdefault(it["prompt"], []).append(it)
    keys = list(groups)[:PROBE] if PROBE else list(groups)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("bench-batch", exist_ok=True)
    print(f"[qi-batch] {len(keys)} 个 prompt / {sum(len(groups[k]) for k in keys)} 张图", flush=True)

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(os.environ["MODEL_PATH"], dtype=torch.bfloat16).to("cuda")
    print(f"[qi-batch] pipeline 加载 {time.time()-t0:.1f}s", flush=True)

    rows, t1 = [], time.time()
    for n, prompt in enumerate(keys, 1):
        group = groups[prompt]
        pp = group[0].get("params") or {}
        W = int(pp.get("width") or 2048); H = int(pp.get("height") or 2048)
        steps = int(pp.get("num_inference_steps") or 8)
        kv = bool(pp.get("use_kv_cache", True)); cfg = float(pp.get("guidance_scale") or 1.0)
        todo = [it for it in group
                if not (RESUME and os.path.exists(os.path.join(OUT, it["id"] + ".png"))
                        and os.path.getsize(os.path.join(OUT, it["id"] + ".png")) > 100 * 1024)]
        if not todo:
            print(f"[qi-batch] {n}/{len(keys)} skip（已有产物）", flush=True)
            continue
        t = time.time()
        try:
            for i in range(0, len(todo), MAX_BATCH):
                chunk = todo[i:i + MAX_BATCH]
                gens = [torch.Generator("cpu").manual_seed(int(it["seed"])) for it in chunk]
                out = pipe(prompt=[prompt] * len(chunk), width=W, height=H,
                           num_inference_steps=steps, true_cfg_scale=cfg, use_kv_cache=kv,
                           generator=gens, num_images_per_prompt=1).images
                for it, img in zip(chunk, out):
                    p = os.path.join(OUT, it["id"] + ".png")
                    img.save(p, format="PNG")
                    rows.append({"id": it["id"], "batch": len(chunk), "sec": round(time.time() - t, 2)})
            print(f"[qi-batch] {n}/{len(keys)} OK batch≤{MAX_BATCH} {time.time()-t:.1f}s "
                  f"({(time.time()-t)/len(todo):.1f}s/张)", flush=True)
        except Exception as e:
            print(f"[qi-batch] {n}/{len(keys)} FAIL {type(e).__name__}: {str(e)[:160]}", flush=True)
            rows.append({"id": group[0]["id"], "batch": len(todo), "error": str(e)[:160]})

    el = time.time() - t1
    ok = [r for r in rows if "sec" in r]
    summary = {"prompts": len(keys), "images": len(ok), "elapsed_s": round(el, 1),
               "per_image_s": round(sum(r["sec"] for r in ok) / len(ok), 2) if ok else None,
               "batches": sorted({r["batch"] for r in ok})}
    json.dump({"summary": summary, "rows": rows}, open("bench-batch/result.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
