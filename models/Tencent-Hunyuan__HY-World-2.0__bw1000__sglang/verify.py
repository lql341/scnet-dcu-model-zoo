#!/usr/bin/env python3
# 精度验证：官方 T03 明示不评分，且未下发 DTU 评价脚本（Chamfer/PSNR）。
# 本脚本验证「产物角色是否齐全」而不是打分。
import json, os, sys

OUT = os.environ.get("OUTDIR", "official-eval")
ROLES = {"scene": ["points.ply", "gaussians.ply"],
         "cameras": ["camera_params.json"],
         "render": ["renders", "render_mosaic.png"]}


def main():
    stats_path = os.path.join(OUT, "stats.json")
    if not os.path.exists(stats_path):
        sys.exit(f"[BLOCKED] 缺 {stats_path}（先跑 container/submit.slurm）")
    stats = json.load(open(stats_path, encoding="utf-8"))
    bad = [k for k, v in stats.items() if not v.get("renders")]
    print(f"[verify] 场景 {len(stats)} 个，渲染缺失 {len(bad)} 个，形状校验见 stats.json")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
