#!/usr/bin/env python3
"""AuK 官方条件评测（120 条 = 中英文各 20 参考音 × 3 seed）。

官方约束：
  - 输入 `assets.reference_audio` + `ref_text` + 目标 `text`/`input` + 保留相关 `instructions`
  - 输出 24 kHz 单声道语音，按 `<id>.wav` 命名
  - **不进行统一评分和性能比较**
"""
import json, os, sys, time

def load_items(cond, asset_root):
    base = os.path.dirname(cond)
    items = []
    for line in open(cond, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        rel = r["assets"]["reference_audio"]["path"]
        p = os.path.normpath(os.path.join(base, rel))
        if not os.path.exists(p):
            r2 = rel
            while r2.startswith("../"): r2 = r2[3:]
            p = os.path.normpath(os.path.join(asset_root, r2))
        r["_ref"] = p
        items.append(r)
    return items


def main():
    cond = os.environ.get("COND_FILE", "conditions/auk.jsonl")
    asset_root = os.environ.get("ASSET_ROOT", "/assets")
    ckpt_dir = os.environ["MODEL_PATH"]
    outdir = os.environ.get("OUTDIR", "outputs")
    os.makedirs(outdir, exist_ok=True)

    items = load_items(cond, asset_root)
    probe = os.environ.get("PROBE_ONLY", "0")
    if probe not in ("0", ""):
        items = items[:int(probe)]
        print(f"[auk] PROBE_ONLY={probe}", flush=True)
    miss = [i for i in items if not os.path.exists(i["_ref"])]
    print(f"[auk] 条件 {len(items)} 条，参考音缺失 {len(miss)}", flush=True)

    import torch
    # torchcodec 的 native lib 绑定 stock torch ABI，DCU 版 torch 加载不了
    # （RuntimeError: Could not load libtorchcodec）。改用 soundfile 后端解码。
    # torchcodec 的 native lib 绑定 stock torch ABI，DCU 版 torch 加载不了；
    # 且本版 torchaudio 的 "soundfile" 后端也不可用。直接用 soundfile 解码，
    # 完全绕开 torchaudio.load（返回 (channels, samples) 的 float32 张量，与 torchaudio 语义一致）。
    import torchaudio
    import numpy as _np, soundfile as _sf

    def _load_compat(uri, *a, **kw):
        data, sr = _sf.read(str(uri), dtype="float32", always_2d=True)
        return torch.from_numpy(_np.ascontiguousarray(data.T)), int(sr)

    torchaudio.load = _load_compat
    print("[auk] 音频解码: soundfile 直读（绕过 torchcodec/torchaudio.load）", flush=True)

    from auk.infer.infer_auk import AukInfer
    print(f"[auk] torch {torch.__version__} cuda {torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    eng = AukInfer(os.path.join(ckpt_dir, "config.yaml"),
                   os.path.join(ckpt_dir, "auk_base.safetensors"),
                   device="cuda:0", dtype="bf16",
                   qwen_path=os.environ.get("ENCODER_PATH"))
    print(f"[auk] 引擎加载 {time.time()-t0:.1f}s", flush=True)

    res, t1 = [], time.time()
    for n, it in enumerate(items, 1):
        out_expect = os.path.join(outdir, it["id"] + ".wav")
        # 【2026-11-03】单条要 ~150 s（nfe=32、6 s 音频），120 条 ≈ 5 h，一趟作业跑不完。
        # 已存在的产物直接计为成功，改作业墙钟后可直接续跑，不重算。
        if os.path.exists(out_expect) and os.path.getsize(out_expect) > 1024:
            res.append({"id": it["id"], "status": "成功", "path": out_expect,
                        "sr": 24000, "dur": None, "resumed": True})
            if n % 5 == 0 or n == len(items):
                print(f"[auk] {n}/{len(items)}  resume")
            continue
        try:
            # 官方要求的输入映射：参考音 + ref_text + 目标 text + instructions
            instruction = it.get("instructions") or ""
            target = it.get("text") or it.get("input") or ""
            msgs = [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [
                    {"type": "audio", "audio": it["_ref"]},
                    {"type": "text", "text": f"{instruction}\n{target}"},
                ]},
            ]
            p = it["params"]
            wav, sr = eng.generate(
                msgs, audio=it["_ref"],
                gen_seconds=float(p.get("gen_seconds") or 6.0),
                nfe=int(p.get("nfe", 32)),
                cfg_strength=float(p.get("cfg_strength", 2.0)),
                sway_sampling_coef=float(p.get("sway_sampling_coef", -1.0)),
                seed=int(it.get("seed") or 42),
            )
            out = os.path.join(outdir, it["id"] + ".wav")
            # 【2026-11-03】torchaudio.save 走 torchcodec 编码器，DCU 版 torch 加载不了
            # libtorchcodec（报 RuntimeError: Could not load libtorchcodec），
            # 与 torchaudio.load 同一个问题。写盘改用 soundfile，全程不碰 torchcodec。
            import soundfile as _sfw
            _arr = wav.detach().float().cpu().numpy() if hasattr(wav, "detach") else _np.asarray(wav)
            if _arr.ndim > 1:
                _arr = _arr.squeeze()
            _sfw.write(out, _arr, int(sr), subtype="PCM_16")
            res.append({"id": it["id"], "status": "成功", "path": out,
                        "sr": sr, "dur": round(wav.shape[-1] / sr, 2)})
        except Exception as e:
            res.append({"id": it["id"], "status": "失败",
                        "error": f"{type(e).__name__}: {str(e)[:200]}"})
        if n % 5 == 0 or n == len(items):
            el = time.time() - t1
            print(f"[auk] {n}/{len(items)}  {el:.0f}s  eta={el/n*(len(items)-n):.0f}s", flush=True)

    with open("反馈表.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in res:
            if r["status"] == "成功":
                f.write(f'{r["id"]},auk,成功,{r["path"]},,无\n')
            else:
                f.write(f'{r["id"]},auk,失败,,{(r.get("error") or "").replace(",","；")},无\n')

    ok = [r for r in res if r["status"] == "成功"]
    summary = {"total": len(res), "ok": len(ok), "fail": len(res)-len(ok),
               "elapsed_s": round(time.time()-t1, 1),
               "sample_rate": 24000,
               "note": "官方不进行统一评分和性能比较；本文件不含评分"}
    json.dump(summary, open("eval_result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
