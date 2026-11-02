#!/usr/bin/env python3
"""Qwen3-ForcedAligner-0.6B 官方条件评测（30 条英文词级对齐）。

官方约束（T03_T04 材料）：
  - 输入：assets.audio + 正确 text + language
  - 输出：每个单词的开始和结束时间，注明单位
  - **BOUNDARIES_PENDING**：没有人工词边界，官方明示「仅收集词级起止时间预测，不计算边界准确率」
  - 不进行统一评分和性能比较

因此本脚本只产出预测结果，不做任何准确率评分（避免伪造指标）。
"""
import json, os, sys, time

def load_items(cond_file, asset_root):
    items = []
    base = os.path.dirname(cond_file)
    for line in open(cond_file, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rel = r["assets"]["audio"]["path"]
        ap = os.path.normpath(os.path.join(base, rel))
        if not os.path.exists(ap):
            r2 = rel
            while r2.startswith("../"):
                r2 = r2[3:]
            ap = os.path.normpath(os.path.join(asset_root, r2))
        if not os.path.exists(ap):
            b = os.path.basename(rel)
            for dp, _, fs in os.walk(asset_root):
                if b in fs:
                    ap = os.path.join(dp, b); break
        r["_audio"] = ap
        items.append(r)
    return items


def main():
    cond = os.environ.get("COND_FILE", "conditions/forced_aligner.jsonl")
    asset_root = os.environ.get("ASSET_ROOT", "/assets")
    model_path = os.environ["MODEL_PATH"]
    outdir = os.environ.get("OUTDIR", "outputs")
    os.makedirs(outdir, exist_ok=True)

    items = load_items(cond, asset_root)
    missing = sum(1 for i in items if not os.path.exists(i["_audio"]))
    print(f"[align] 条件 {len(items)} 条，素材缺失 {missing}", flush=True)

    import torch
    from qwen_asr import Qwen3ForcedAligner
    print(f"[align] torch {torch.__version__} | cuda avail {torch.cuda.is_available()} | "
          f"devs {torch.cuda.device_count()}", flush=True)

    t0 = time.time()
    model = Qwen3ForcedAligner.from_pretrained(model_path, dtype=torch.bfloat16, device_map="cuda:0")
    print(f"[align] 模型加载 {time.time()-t0:.1f}s", flush=True)

    results, t1 = [], time.time()
    for n, it in enumerate(items, 1):
        try:
            res = model.align(audio=it["_audio"], text=it["text"],
                              language=it["params"].get("language") or "English")
            words = [{"text": w.text, "start": round(w.start_time, 3), "end": round(w.end_time, 3)}
                     for w in res[0]]
            with open(os.path.join(outdir, it["id"] + ".json"), "w", encoding="utf-8") as f:
                json.dump({"id": it["id"], "unit": "seconds", "words": words}, f,
                          ensure_ascii=False, indent=2)
            # 同时存一份纯文本（便于人工查看）
            with open(os.path.join(outdir, it["id"] + ".tsv"), "w", encoding="utf-8") as f:
                f.write("word\tstart_s\tend_s\n")
                for w in words:
                    f.write(f'{w["text"]}\t{w["start"]}\t{w["end"]}\n')
            results.append({"id": it["id"], "status": "成功", "n_words": len(words),
                            "ref_words": len(it["text"].split()),
                            "dur": (words[-1]["end"] if words else 0)})
        except Exception as e:
            results.append({"id": it["id"], "status": "失败",
                            "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if n % 5 == 0 or n == len(items):
            el = time.time() - t1
            print(f"[align] {n}/{len(items)}  {el:.0f}s  eta={el/n*(len(items)-n):.0f}s", flush=True)

    with open("反馈表.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in results:
            if r["status"] == "成功":
                f.write(f'{r["id"]},forced_aligner,成功,{outdir}/{r["id"]}.json,,无\n')
            else:
                f.write(f'{r["id"]},forced_aligner,失败,,{(r.get("error") or "").replace(",","；")},无\n')

    ok = [r for r in results if r["status"] == "成功"]
    summary = {
        "total": len(results), "ok": len(ok),
        "fail": len(results) - len(ok),
        "evaluation_status": "BOUNDARIES_PENDING",
        "note": "官方明示：无人工词边界，仅收集词级起止时间预测，不计算边界准确率；本文件不含任何准确率数字",
        "word_count": {"predicted_total": sum(r["n_words"] for r in ok),
                       "reference_total": sum(r["ref_words"] for r in ok)},
        "elapsed_s": round(time.time() - t1, 1),
    }
    json.dump(summary, open("eval_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))



if __name__ == "__main__":
    main()
