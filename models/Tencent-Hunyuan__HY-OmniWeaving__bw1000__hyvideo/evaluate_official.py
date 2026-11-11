#!/usr/bin/env python3
"""HY-OmniWeaving 官方条件评测（20 prompt × 3 seed = 60 条，832×480 / 81 帧 / 16 fps）。

官方约束（conditions/omniweaving.jsonl）：
  - 输入 `prompt`（文本）
  - 固定条件：832×480、81 帧、16 fps、50 步、guidance 6.0、flow_shift 7.0、think=False、bf16
  - 产物按 `<id>.mp4` 命名
  - 官方明示「不进行统一评分和性能比较」→ 本脚本不计算任何指标

并行方式：官方 generate.py 用 `WORLD_SIZE` 做序列并行（`initialize_parallel_state(sp=WORLD_SIZE)`），
因此用 `torchrun --nproc_per_node=N` 起 N 卡协同跑同一条视频（N=1 即单卡）。

续跑：产物已存在且 > 200 KB 的条目直接跳过（81 帧 480p 的 mp4 不会小于这个数）。
"""
import json, os, struct, subprocess, sys, time

MIN_BYTES = 40 * 1024          # 低于这个基本是空文件/截断
EXPECT_DUR = 81 / 16           # 5.0625 s，官方 81 帧 / 16 fps


def _boxes(buf, start, end):
    i = start
    while i + 8 <= end:
        size = struct.unpack(">I", buf[i:i + 4])[0]
        typ = buf[i + 4:i + 8]
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", buf[i + 8:i + 16])[0]; hdr = 16
        if size < hdr:
            return
        yield typ, i + hdr, i + size
        i += size


def mp4_ok(path):
    """真正校验产物：能解析出 moov/mvhd，且时长≈5.06 s。
    为什么不用文件大小：81 帧 832×480 的 H.264 依内容可从 100 KB 到 500 KB，
    早先按 200 KB 判会把手头已有的合法产物误判成无效并重跑（2026-11-11 踩过，白烧 2 h）。"""
    try:
        if os.path.getsize(path) < MIN_BYTES:
            return False
        with open(path, "rb") as f:
            buf = f.read(4 << 20)          # moov 一般在头部；少数在尾部，退化为只看大小
        for typ, s_, e_ in _boxes(buf, 0, len(buf)):
            if typ != b"moov":
                continue
            for t2, s2, e2 in _boxes(buf, s_, e_):
                if t2 != b"mvhd":
                    continue
                ver = buf[s2]
                if ver == 0:
                    ts, dur = struct.unpack(">II", buf[s2 + 12:s2 + 20])
                else:
                    ts = struct.unpack(">I", buf[s2 + 20:s2 + 24])[0]
                    dur = struct.unpack(">Q", buf[s2 + 24:s2 + 32])[0]
                return ts > 0 and abs(dur / ts - EXPECT_DUR) < 0.3
        return os.path.getsize(path) > MIN_BYTES   # 解析不到 moov 时按大小兜底
    except Exception:
        return False


def load_items(cond):
    with open(cond, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    cond = os.environ.get("COND_FILE", "conditions/omniweaving.jsonl")
    model = os.environ["MODEL_PATH"]
    src = os.environ.get("OMNI_SRC", "/opt/omni-src")
    outdir = os.environ.get("OUTDIR", "outputs")
    nproc = int(os.environ.get("NPROC", "1"))
    os.makedirs(outdir, exist_ok=True)
    run_tag = os.environ.get("RUN_TAG", "") or str(os.getpid())
    os.makedirs("logs/shard", exist_ok=True)

    items = load_items(cond)
    probe = int(os.environ.get("PROBE_ONLY", "0") or 0)
    if probe > 0:
        items = items[:probe]
        print(f"[omni] PROBE_ONLY={probe}", flush=True)
    print(f"[omni] 条件 {len(items)} 条；nproc={nproc}", flush=True)

    rows, t0 = [], time.time()
    for n, it in enumerate(items, 1):
        out = os.path.abspath(os.path.join(outdir, it["id"] + ".mp4"))
        if os.path.exists(out) and mp4_ok(out):
            rows.append({"id": it["id"], "status": "成功", "path": out,
                         "bytes": os.path.getsize(out), "resumed": True})
            print(f"[omni] {n}/{len(items)} resume {it['id']}", flush=True)
            continue
        p = it.get("params") or {}
        cmd = ["generate.py",
               "--prompt", it["prompt"],
               "--model_path", model,
               "--task", "t2v",
               "--resolution", "480p",
               # 【关键】官方条件是 832×480；该 pipeline 按"比例"从桶列表里挑尺寸，
               # 传 16:9（0.5625）会落到 848×480 桶。传 "832:480" 才能精确命中 (832,480) 桶。
               # 实测踩过：用 16:9 跑完 60 条，产物全是 848×480（宽 +2%）。
               "--aspect_ratio", f"{int(p.get('width', 832))}:{int(p.get('height', 480))}",
               "--video_length", str(int(p.get("num_frames", 81))),
               "--num_inference_steps", str(int(p.get("num_inference_steps", 50))),
               "--total_steps", str(int(p.get("num_inference_steps", 50))),
               "--seed", str(int(it.get("seed") or 42)),
               "--dtype", str(p.get("dtype", "bf16")),
               "--output_path", out]
        if not bool(p.get("think", False)):
            pass                     # 官方条件 think=False，即不传 --think
        else:
            cmd.append("--think")
        if nproc > 1:
            cmd = ["torchrun", f"--nproc_per_node={nproc}", "--master_port",
                   str(29500 + os.getpid() % 400)] + cmd
        else:
            cmd = [sys.executable] + cmd

        t1 = time.time()
        elog = f"logs/shard/stderr_{it['id']}_{run_tag}.log"
        with open(elog, "w", encoding="utf-8") as fe:
            r = subprocess.run(cmd, cwd=src, text=True, timeout=8 * 3600,
                               stdout=fe, stderr=subprocess.STDOUT)
        ok = r.returncode == 0 and os.path.exists(out) and os.path.getsize(out) > TOL_BYTES
        if ok:
            rows.append({"id": it["id"], "status": "成功", "path": out,
                         "bytes": os.path.getsize(out), "sec": round(time.time() - t1, 1)})
            print(f"[omni] {n}/{len(items)} OK {it['id']}  {time.time()-t1:.0f}s "
                  f"{(os.path.getsize(out)/1e6):.1f}MB", flush=True)
        else:
            tail = [l for l in open(elog, encoding="utf-8", errors="replace").read().splitlines()
                    if any(k in l for k in ("Error", "error", "Traceback", "OOM", "out of memory"))]
            msg = (tail[-1][:200] if tail else f"rc={r.returncode}")
            rows.append({"id": it["id"], "status": "失败", "error": msg})
            print(f"[omni] {n}/{len(items)} FAIL {it['id']}  {msg}", flush=True)

    with open(f"反馈表-{run_tag}.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for row in rows:
            if row["status"] == "成功":
                f.write(f'{row["id"]},omniweaving,成功,{row["path"]},,无\n')
            else:
                f.write(f'{row["id"]},omniweaving,失败,,{(row.get("error") or "").replace(",", "；")},无\n')

    ok = [r for r in rows if r["status"] == "成功"]
    summary = {"total": len(rows), "ok": len(ok), "fail": len(rows) - len(ok),
               "elapsed_s": round(time.time() - t0, 1), "nproc": nproc,
               "bytes_total_MB": round(sum(r.get("bytes", 0) for r in ok) / 1e6, 1),
               "note": "官方不进行统一评分和性能比较；本文件不含评分"}
    json.dump(summary, open(f"eval_result-{run_tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
