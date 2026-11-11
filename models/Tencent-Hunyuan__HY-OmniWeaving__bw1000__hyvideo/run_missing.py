#!/usr/bin/env python3
"""只跑「还没产物」的官方条件，直接调官方 generate.py，不写任何汇总文件。

为什么不用 evaluate_official.py：在本集群的 NFS 上，容器（rootless podman，root 映射成 subuid）
**能创建新文件、但不能覆盖已存在的文件**（哪怕 666 也不行）。汇总文件（反馈表/stats）必然要覆盖，
所以把它们挪到宿主侧做。这个脚本只做一件事：为缺失的 id 生成 mp4（都是新建文件）。
"""
import json, os, subprocess, sys, time

COND = os.environ.get("COND_FILE", "conditions/omniweaving.jsonl")
OUT = os.environ.get("OUTDIR", "official-eval/outputs")
SRC = os.environ.get("OMNI_SRC", "/opt/omni-src")
MODEL = os.environ["MODEL_PATH"]
NPROC = int(os.environ.get("NPROC", "1"))

def main():
    items = [json.loads(l) for l in open(COND, encoding="utf-8") if l.strip()]
    os.makedirs(OUT, exist_ok=True)
    todo = [it for it in items if not os.path.exists(os.path.join(OUT, it["id"] + ".mp4"))]
    probe = int(os.environ.get("PROBE_ONLY", "0") or 0)
    if probe:
        todo = todo[:probe]
    print(f"[miss] 总 {len(items)}，待跑 {len(todo)}", flush=True)
    ok = fail = 0
    for n, it in enumerate(todo, 1):
        p = it.get("params") or {}
        out = os.path.abspath(os.path.join(OUT, it["id"] + ".mp4"))
        cmd = ["generate.py", "--prompt", it["prompt"], "--model_path", MODEL,
               "--task", "t2v", "--resolution", "480p",
               "--aspect_ratio", f"{int(p.get('width',832))}:{int(p.get('height',480))}",
               "--video_length", str(int(p.get("num_frames", 81))),
               "--num_inference_steps", str(int(p.get("num_inference_steps", 50))),
               "--total_steps", str(int(p.get("num_inference_steps", 50))),
               "--seed", str(int(it.get("seed") or 42)),
               "--dtype", str(p.get("dtype", "bf16")),
               "--output_path", out]
        if NPROC > 1:
            cmd = ["torchrun", f"--nproc_per_node={NPROC}"] + cmd
        else:
            cmd = [sys.executable] + cmd
        t0 = time.time()
        log = f"/tmp/gen-{it['id']}.log"
        with open(log, "w", encoding="utf-8") as fe:
            rc = subprocess.run(cmd, cwd=SRC, stdout=fe, stderr=subprocess.STDOUT).returncode
        size = os.path.getsize(out) if os.path.exists(out) else 0
        if rc == 0 and size > 40 * 1024:
            ok += 1
            print(f"[miss] {n}/{len(todo)} OK {it['id']} {time.time()-t0:.0f}s {size/1e6:.2f}MB", flush=True)
        else:
            fail += 1
            tail = [l for l in open(log, encoding="utf-8", errors="replace").read().splitlines()[-6:]]
            print(f"[miss] {n}/{len(todo)} FAIL {it['id']} rc={rc} size={size}", flush=True)
            for l in tail: print("   |", l[:200], flush=True)
    print(f"[miss] 完成 ok={ok} fail={fail}", flush=True)

if __name__ == "__main__":
    main()
