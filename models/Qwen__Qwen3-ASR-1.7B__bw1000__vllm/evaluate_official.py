#!/usr/bin/env python3
"""按官方条件清单执行 ASR 评测，输出 <id>.txt + 反馈表.csv + 环境说明.txt，并自算 CER/WER。

官方约束（见 official/t03-t04/T03_T04模型可行性方案_框架数据接入评测.md）：
  - 只把 assets.audio 与 params 输入模型，**不传参考 text，不传 language 标签**
  - temperature=0（greedy）、max_new_tokens=1024、language=null（自动识别）、dtype=bf16
  - 输出 UTF-8 文本，文件名为该条 id
  - 不得把数据集里的参考答案当作测试结果

自算 CER/WER 仅用于本仓库记录，官方不要求评分、也不做统一评分。
"""
import io, json, os, re, sys, time, wave, base64, unicodedata, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

COND = os.environ.get("COND_FILE", "conditions/qwen3_asr.jsonl")
OUTDIR = os.environ.get("OUTDIR", "outputs")
PORT = os.environ.get("PORT", "30000")
MODEL = os.environ.get("SERVED_NAME", "qwen3-asr-1.7b")
WORKERS = int(os.environ.get("WORKERS", "4"))
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_items():
    items = []
    with open(COND, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rel = r["assets"]["audio"]["path"]
            ap = os.path.normpath(os.path.join(os.path.dirname(COND), rel))
            if not os.path.exists(ap):
                # 官方条件文件里的 assets 路径是相对 formal/conditions/ 的 ../../data/...
                # 素材被单独挂在 ASSET_ROOT 下时，逐级剥掉 ../ 再拼接
                root = os.environ.get("ASSET_ROOT", ".")
                r2 = rel
                while r2.startswith("../"):
                    r2 = r2[3:]
                ap = os.path.normpath(os.path.join(root, r2))
            if not os.path.exists(ap):
                # 最后兜底：按 basename 在 ASSET_ROOT 下找
                base = os.path.basename(rel)
                for dirpath, _, files in os.walk(os.environ.get("ASSET_ROOT", ".")):
                    if base in files:
                        ap = os.path.join(dirpath, base)
                        break
            r["_audio"] = ap
            items.append(r)
    return items


def transcribe(item):
    """按官方参数调用；只传音频。"""
    b64 = base64.b64encode(open(item["_audio"], "rb").read()).decode()
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + b64}}]}],
        "temperature": item["params"]["temperature"],          # 0 = greedy
        "max_tokens": item["params"]["max_new_tokens"],        # 1024
        # 不传 language：官方要求自动识别
    }
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=1800) as r:
        d = json.load(r)
    text = d["choices"][0]["message"].get("content") or ""
    # 剥掉 "language Chinese<asr_text>" 前缀，只留识别文本
    m = re.search(r"<asr_text>(.*)", text, re.S)
    if m:
        text = m.group(1)
    return text.strip(), (d.get("usage") or {})


def norm_zh(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s，。、？！；：\"'（）()\-,.!?;: ]+", "", s)


def norm_en(s):
    s = unicodedata.normalize("NFKC", s).upper()
    s = re.sub(r"[^A-Z' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        prev, dp[0] = dp[0], i
        for j, hc in enumerate(hyp, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return dp[len(hyp)] / len(ref)


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    dp = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, dp[0] = dp[0], i
        for j, hw in enumerate(h, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return dp[len(h)] / len(r)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    items = load_items()
    print(f"[eval] 条件清单 {COND}：{len(items)} 条", flush=True)

    results = [None] * len(items)
    t0 = time.time()

    def work(i):
        it = items[i]
        try:
            text, usage = transcribe(it)
            with open(os.path.join(OUTDIR, it["id"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            results[i] = {"id": it["id"], "status": "成功", "text": text,
                          "ref": it.get("text"), "lang": it.get("language"),
                          "source": it.get("source", "").split(":")[0], "usage": usage}
        except Exception as e:
            results[i] = {"id": it["id"], "status": "失败", "error":
                          f"{type(e).__name__}: {str(e)[:200]}", "ref": it.get("text"),
                          "lang": it.get("language"), "source": it.get("source", "").split(":")[0]}
        if (i + 1) % 20 == 0 or i + 1 == len(items):
            el = time.time() - t0
            print(f"[eval] {i+1}/{len(items)}  elapsed={el:.0f}s  eta={el/(i+1)*(len(items)-i-1):.0f}s", flush=True)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, range(len(items))))

    # 反馈表
    with open("反馈表.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in results:
            if r["status"] == "成功":
                f.write(f'{r["id"]},qwen3_asr,成功,{OUTDIR}/{r["id"]}.txt,,无\n')
            else:
                err = (r.get("error") or "").replace(",", "；")
                f.write(f'{r["id"]},qwen3_asr,失败,,{err},无\n')

    # 自算指标（官方不要求）
    agg = {}
    for r in results:
        if r["status"] != "成功" or not r.get("ref"):
            continue
        g = r["source"] or r["lang"] or "?"
        a = agg.setdefault(g, {"n": 0, "cer": 0.0, "wer": 0.0, "chars": 0})
        if r["lang"] == "zh":
            ref, hyp = norm_zh(r["ref"]), norm_zh(r["text"])
            a["cer"] += cer(ref, hyp) * len(ref); a["chars"] += len(ref); a["n"] += 1
        else:
            ref, hyp = norm_en(r["ref"]), norm_en(r["text"])
            a["cer"] += cer(ref, hyp) * len(ref); a["chars"] += len(ref)
            a["wer"] += wer(ref, hyp); a["n"] += 1

    summary = {"total": len(results),
               "ok": sum(1 for r in results if r["status"] == "成功"),
               "fail": sum(1 for r in results if r["status"] != "成功"),
               "groups": {g: {"n": v["n"],
                              "CER": round(100*v["cer"]/v["chars"], 2) if v["chars"] else None,
                              "WER": round(100*v["wer"]/v["n"], 2)} for g, v in agg.items()}}
    with open("eval_result.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 环境说明
    import platform, subprocess
    try:
        import torch
        dev = f"{torch.cuda.get_device_name(0)} × {torch.cuda.device_count()}"
        torchv = torch.__version__
    except Exception:
        dev, torchv = "unknown", "unknown"
    try:
        import vllm
        vllmv = vllm.__version__
    except Exception:
        vllmv = "unknown"
    with open("环境说明.txt", "w", encoding="utf-8") as f:
        f.write(f"""模型及权重版本：{os.environ.get('MODEL_ID','Qwen/Qwen3-ASR-1.7B')}（ModelScope 快照，未指定 revision）
框架及版本：vLLM {vllmv}
torch 版本：{torchv}
设备信息：{dev}
镜像：{os.environ.get('IMAGE','unknown')}
执行范围：{COND}，共 {len(results)} 条（id 见 反馈表.csv）
解码参数：temperature=0（greedy）、max_new_tokens=1024、language=null（自动识别）、dtype=bf16
说明：输入仅使用条件清单的 assets.audio，未传入参考 text 与 language 标签。
      CER/WER 为本仓库自算（见 eval_result.json），官方不要求评分。
""")


if __name__ == "__main__":
    main()
