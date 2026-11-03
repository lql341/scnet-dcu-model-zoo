#!/usr/bin/env python3
"""Qwen-Image-2.1-Turbo 官方条件评测（60 输入 × 3 seed = 180 条，输出 2048×2048）。

官方约束：
  - 使用 `prompt` 作为输入，读取对应 seed 及参数
  - 输出尺寸固定 2048×2048
  - 产物按 id 命名：`<id>.png`
  - 固定 seed 42 / 2026 / 1234
"""
import json, os, sys, time

def load_items(cond):
    items = []
    for line in open(cond, encoding="utf-8"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def main():
    cond = os.environ.get("COND_FILE", "conditions/qwen_image.jsonl")
    outdir = os.environ.get("OUTDIR", "outputs")
    os.makedirs(outdir, exist_ok=True)
    items = load_items(cond)
    probe = os.environ.get("PROBE_ONLY", "0")
    if probe not in ("0", ""):
        items = items[:int(probe)]
        print(f"[qi] PROBE_ONLY={probe}，只跑前 {len(items)} 条做管线验证", flush=True)
    print(f"[qi] 条件 {len(items)} 条", flush=True)

    import torch
    from diffusers import QwenImage21Pipeline
    print(f"[qi] torch {torch.__version__} cuda {torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    pipe = QwenImage21Pipeline.from_pretrained(os.environ["MODEL_PATH"], dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    print(f"[qi] pipeline 加载 {time.time()-t0:.1f}s", flush=True)

    W0 = int(os.environ.get("WIDTH", 2048)); H0 = int(os.environ.get("HEIGHT", 2048))
    res, t1 = [], time.time()
    for n, it in enumerate(items, 1):
        try:
            g = torch.Generator("cpu").manual_seed(int(it["seed"]))
            # 【2026-11-03】直接取条件清单里的 params（官方给的就是 width/height=2048、
            # num_inference_steps=8、use_kv_cache=true）。之前写死 use_kv_cache=True 且
            # 未传 num_inference_steps，实际按 pipeline 默认 40 步跑，与官方条件不符。
            pp = it.get("params") or {}
            W = int(pp.get("width") or W0); H = int(pp.get("height") or H0)
            steps = int(pp.get("num_inference_steps") or 8)
            kv = bool(pp.get("use_kv_cache", True))
            cfg = float(pp.get("guidance_scale") or 1.0)
            img = pipe(prompt=it["prompt"], width=W, height=H, num_inference_steps=steps,
                       true_cfg_scale=cfg, use_kv_cache=kv, generator=g).images[0]
            p = os.path.join(outdir, f"{it['id']}.png")
            img.save(p, format="PNG")
            res.append({"id": it["id"], "status": "成功", "path": p,
                        "size": list(img.size), "seed": it["seed"]})
        except Exception as e:
            res.append({"id": it["id"], "status": "失败",
                        "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if n % 10 == 0 or n == len(items):
            el = time.time() - t1
            print(f"[qi] {n}/{len(items)}  {el:.0f}s  eta={el/n*(len(items)-n):.0f}s", flush=True)

    with open("反馈表.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in res:
            if r["status"] == "成功":
                f.write(f'{r["id"]},qwen_image,成功,{r["path"]},,无\n')
            else:
                f.write(f'{r["id"]},qwen_image,失败,,{(r.get("error") or "").replace(",","；")},无\n')

    ok = [r for r in res if r["status"] == "成功"]
    summary = {"total": len(res), "ok": len(ok), "fail": len(res)-len(ok),
               "width": W, "height": H,
               "elapsed_s": round(time.time()-t1, 1),
               "note": "官方固定条件 2048×2048；本文件不含评分（官方不进行统一评分）"}
    json.dump(summary, open("eval_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
