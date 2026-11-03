#!/usr/bin/env python3
"""Wan2.1-I2V-14B-480P 官方条件评测（20 张首帧 × 3 seed = 60 条，832×480 / 81 帧 / 16 fps）。

官方约束（conditions/wan_i2v.jsonl）：
  - 输入 `assets.image`（首帧）+ `prompt` + `negative_prompt`
  - 固定条件：832×480、81 帧、16 fps、40 步、unipc、shift 3.0、guidance 5.0、bf16
  - 产物按 `<id>.mp4` 命名
  - 官方明示「不进行统一评分和性能比较」→ 本脚本不计算任何指标

并行：本脚本支持分片（SHARD_INDEX / SHARD_TOTAL），由 submit.slurm 按申请的卡数设置，
1 卡 1 进程、每进程一张卡（HIP_VISIBLE_DEVICES 由 Slurm 给）。

续跑：产物已存在且 > 1 MB 的条目直接跳过（81 帧 480p 的 mp4 不可能小于 1 MB）。
"""
import json, os, subprocess, sys, time

TOL_BYTES = 1 << 20


def load_items(cond):
    items = []
    with open(cond, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolve_image(it, asset_root, cond_dir):
    rel = it["assets"]["image"]["path"]
    p = os.path.normpath(os.path.join(cond_dir, rel))
    if not os.path.exists(p):
        r = rel
        while r.startswith("../"):
            r = r[3:]
        p = os.path.normpath(os.path.join(asset_root, r))
    return p


def main():
    cond = os.environ.get("COND_FILE", "conditions/wan_i2v.jsonl")
    asset_root = os.environ.get("ASSET_ROOT", "/assets")
    ckpt = os.environ["MODEL_PATH"]
    src = os.environ.get("WAN_SRC", "/wan-src")
    outdir = os.environ.get("OUTDIR", "outputs")
    task = os.environ.get("TASK", "i2v-14B")
    shard = int(os.environ.get("SHARD_INDEX", "0"))
    total = int(os.environ.get("SHARD_TOTAL", "1"))
    os.makedirs(outdir, exist_ok=True)

    items = load_items(cond)
    mine = [it for i, it in enumerate(items) if i % total == shard]
    probe = int(os.environ.get("PROBE_ONLY", "0") or 0)
    if probe > 0:
        mine = mine[:probe]
        print(f"[wan] PROBE_ONLY={probe}：只跑本分片前 {len(mine)} 条", flush=True)
    print(f"[wan] 条件 {len(items)} 条；本分片 {shard}/{total} → {len(mine)} 条", flush=True)

    miss = [it["id"] for it in mine if not os.path.exists(resolve_image(it, asset_root, os.path.dirname(cond)))]
    if miss:
        print(f"[wan] 首帧缺失 {len(miss)} 条：{miss[:3]}", flush=True)

    rows, t0 = [], time.time()
    for n, it in enumerate(mine, 1):
        out = os.path.join(outdir, it["id"] + ".mp4")
        if os.path.exists(out) and os.path.getsize(out) > TOL_BYTES:
            rows.append({"id": it["id"], "status": "成功", "path": out, "bytes": os.path.getsize(out), "resumed": True})
            print(f"[wan] {n}/{len(mine)} resume {it['id']}", flush=True)
            continue
        p = it.get("params") or {}
        img = resolve_image(it, asset_root, os.path.dirname(cond))
        fsdp = os.environ.get("FSDP", "0") == "1"
        nproc = int(os.environ.get("FSDP_NPROC", "8"))
        # 单卡 64 GB 装不下 14B + UMT5-XXL（实测 --offload_model 与 --t5_cpu 仍 OOM）：
        # 改用官方推荐的多卡 FSDP（--dit_fsdp --t5_fsdp，不需要 xfuser，非序列并行）
        launcher = ["torchrun", f"--nproc_per_node={nproc}", "--master_port", str(29500 + (os.getpid() % 500))]
        cmd = launcher + [
            "generate.py",
            "--task", task,
            "--size", f"{int(p.get('width', 832))}*{int(p.get('height', 480))}",
            "--ckpt_dir", ckpt,
            "--image", img,
            "--prompt", it["prompt"],
            "--negative_prompt", it.get("negative_prompt") or "",
            "--frame_num", str(int(p.get("num_frames", 81))),
            "--sample_steps", str(int(p.get("num_inference_steps", 40))),
            "--sample_guide_scale", str(float(p.get("guidance_scale", 5.0))),
            "--sample_shift", str(float(p.get("sample_shift", 3.0))),
            "--sample_solver", str(p.get("sample_solver", "unipc")),
            "--base_seed", str(int(it.get("seed") or 42)),
            "--offload_model", os.environ.get("OFFLOAD_MODEL", "True"),
            "--save_file", out,
        ]
        # 显存：14B bf16 + UMT5-XXL 在 64 GB 单卡上会 OOM（实测 Offloading 仍打满 63.98 GB），
        # 把 T5 放 CPU 可省约 11 GB；再不够就用官方 --dit_fsdp + torchrun 多卡切分。
        if os.environ.get("T5_CPU", "0") == "1":
            cmd.append("--t5_cpu")
        if fsdp:
            cmd += ["--dit_fsdp", "--t5_fsdp"]
        t1 = time.time()
        try:
            r = subprocess.run(cmd, cwd=src, capture_output=True, text=True, timeout=6 * 3600)
            if r.returncode != 0 or not (os.path.exists(out) and os.path.getsize(out) > TOL_BYTES):
                raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1][:200] if (r.stderr or r.stdout) else f"rc={r.returncode}")
            rows.append({"id": it["id"], "status": "成功", "path": out,
                         "bytes": os.path.getsize(out), "sec": round(time.time() - t1, 1)})
            print(f"[wan] {n}/{len(mine)} OK {it['id']}  {time.time()-t1:.0f}s  {(os.path.getsize(out)/1e6):.1f}MB", flush=True)
        except Exception as e:
            rows.append({"id": it["id"], "status": "失败", "error": f"{type(e).__name__}: {str(e)[:200]}"})
            print(f"[wan] {n}/{len(mine)} FAIL {it['id']}  {type(e).__name__}: {str(e)[:160]}", flush=True)

    with open(f"反馈表-shard{shard}.csv", "w", encoding="utf-8", newline="") as f:
        f.write("condition_id,model,status,output_path,error,parameter_changes\n")
        for r in rows:
            if r["status"] == "成功":
                f.write(f'{r["id"]},wan2.1_i2v,成功,{r["path"]},,无\n')
            else:
                f.write(f'{r["id"]},wan2.1_i2v,失败,,{(r.get("error") or "").replace(",", "；")},无\n')

    ok = [r for r in rows if r["status"] == "成功"]
    print(json.dumps({"shard": shard, "total": len(rows), "ok": len(ok), "fail": len(rows) - len(ok),
                      "elapsed_s": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
