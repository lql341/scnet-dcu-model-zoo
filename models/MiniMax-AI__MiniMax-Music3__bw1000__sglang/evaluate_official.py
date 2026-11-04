#!/usr/bin/env python3
"""MiniMax-Music3 官方条件评测（60 条 = 中英各 10 段歌词 × seed 42/2026/1234）。

官方约束（conditions/music3.jsonl）：
  - 输入 `input`（带段落标签的歌词）+ `instructions`（音乐描述）
  - 固定条件：最多 750 个声学帧、32 kHz 立体声、16 bit
  - 产物按 `<id>.wav` 命名
  - 官方明示「不进行统一评分和性能比较」→ 不计算任何指标

服务：镜像自带的 BW 适配 harness 起 OpenAI 兼容接口 `POST /v1/audio/speech`
（base URL `http://127.0.0.1:$PORT`，请求体见下）。harness 来源见 README §3.1。

续跑：产物已存在且 > 100 KB 的条目直接跳过（30 s 32 kHz 立体声约 3.8 MB）。
"""
import json, os, sys, time, wave, urllib.request, urllib.error

TOL_BYTES = 100 * 1024
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 本机服务，不走代理


def load_items(cond):
    with open(cond, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def synth(port, item):
    p = item.get("params") or {}
    body = {
        "model": os.environ.get("SERVED_NAME", "MiniMaxAI/MiniMax-Music3"),
        "input": item.get("input") or item.get("text") or "",
        "instructions": item.get("instructions") or "",
        "response_format": "wav",
        "seed": int(item.get("seed") or 42),
        "max_new_tokens": int(p.get("max_new_tokens") or 750),
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/audio/speech",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=3600) as r:
        return r.read()


def wav_info(path):
    with wave.open(path) as w:
        return {"sr": w.getframerate(), "ch": w.getnchannels(),
                "bits": 8 * w.getsampwidth(),
                "dur": round(w.getnframes() / w.getframerate(), 2)}


def main():
    cond = os.environ.get("COND_FILE", "conditions/music3.jsonl")
    outdir = os.environ.get("OUTDIR", "outputs")
    port = int(os.environ.get("PORT", "8000"))
    probe = int(os.environ.get("PROBE_ONLY", "0") or 0)
    os.makedirs(outdir, exist_ok=True)

    items = load_items(cond)
    if probe > 0:
        items = items[:probe]
        print(f"[m3] PROBE_ONLY={probe}", flush=True)
    print(f"[m3] 条件 {len(items)} 条；服务端口 {port}", flush=True)

    rows, t0 = [], time.time()
    for n, it in enumerate(items, 1):
        out = os.path.join(outdir, it["id"] + ".wav")
        if os.path.exists(out) and os.path.getsize(out) > TOL_BYTES:
            rows.append({"id": it["id"], "status": "成功", "path": out,
                         "bytes": os.path.getsize(out), "resumed": True})
            print(f"[m3] {n}/{len(items)} resume {it['id']}", flush=True)
            continue
        t1 = time.time()
        try:
            data = synth(port, it)
            if len(data) < TOL_BYTES:
                raise RuntimeError(f"返回体过小（{len(data)} B），疑似错误响应")
            with open(out, "wb") as f:
                f.write(data)
            info = wav_info(out)
            ok_spec = info["sr"] == 32000 and info["ch"] == 2
            rows.append({"id": it["id"], "status": "成功", "path": out,
                         "bytes": len(data), "sec": round(time.time() - t1, 1), **info})
            print(f"[m3] {n}/{len(items)} OK {it['id']}  {time.time()-t1:.0f}s  "
                  f"{info['dur']}s {info['sr']}Hz/{info['ch']}ch"
                  f"{'' if ok_spec else '  ⚠规格不符'}", flush=True)
        except Exception as e:
            detail = ""
            if isinstance(e, urllib.error.HTTPError):
                try:
                    detail = e.read().decode("utf-8", "replace")[:200]
                except Exception:
                    pass
            rows.append({"id": it["id"], "status": "失败",
                         "error": f"{type(e).__name__}: {str(e)[:160]} {detail}".strip()})
            print(f"[m3] {n}/{len(items)} FAIL {it['id']}  {type(e).__name__}: {str(e)[:160]}", flush=True)

    with open("反馈表.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in rows:
            if r["status"] == "成功":
                f.write(f'{r["id"]},minimax_music3,成功,{r["path"]},,无\n')
            else:
                f.write(f'{r["id"]},minimax_music3,失败,,{(r.get("error") or "").replace(",", "；")},无\n')

    ok = [r for r in rows if r["status"] == "成功"]
    durs = [r["dur"] for r in ok if "dur" in r]
    stats = {"total": len(rows), "ok": len(ok), "fail": len(rows) - len(ok),
             "elapsed_s": round(time.time() - t0, 1),
             "spec": {"sample_rate": sorted({r["sr"] for r in ok if "sr" in r}),
                      "channels": sorted({r["ch"] for r in ok if "ch" in r})},
             "dur_s": {"min": min(durs) if durs else None, "max": max(durs) if durs else None,
                       "sum": round(sum(durs), 1) if durs else None},
             "bytes_total_MB": round(sum(r.get("bytes", 0) for r in ok) / 1e6, 1),
             "note": "官方不进行统一评分和性能比较；本文件不含评分"}
    json.dump(stats, open("eval_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
