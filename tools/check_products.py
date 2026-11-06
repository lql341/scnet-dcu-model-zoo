#!/usr/bin/env python3
"""产物规格守门：逐条目核对 official-eval / bench 里的规格统计与 meta.json 是否自相矛盾。

为什么需要它：2026-11-05 的教训 —— HY-OmniWeaving 跑完 60 条后，只检查"文件存在且 >200 KB"
的脚本一路绿到底，实际产物是 848×480（官方要 832×480），26 卡时白烧。
本工具把"入库的 stats.json 必须与条件清单/期望规格一致"变成一道可重复执行的检查。

用法：
  python3 tools/check_products.py            # 全库
  python3 tools/check_products.py models/<条目>/
退出码：0 = 全部一致；1 = 有不一致（会打印明细）
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 官方条件对各模型产物规格的要求（来自 official/t03-t04/conditions/*.jsonl 的 expected 字段）
SPECS = {
    "Qwen-Image-2.1-Turbo": {"kind": "image", "width": 2048, "height": 2048},
    "MiniMax-Music3": {"kind": "audio", "sample_rate": 32000, "channels": 2},
    "AuK": {"kind": "audio", "sample_rate": 24000, "channels": 1},
    "AuK-Flash": {"kind": "audio", "sample_rate": 24000, "channels": 1},
    "Qwen3-ASR-1.7B": {"kind": "text"},
    "Qwen3-ForcedAligner-0.6B": {"kind": "words"},
    "Wan2.1-I2V-14B-480P": {"kind": "video", "width": 832, "height": 480, "frames": 81, "fps": 16},
    "HY-OmniWeaving": {"kind": "video", "width": 832, "height": 480, "frames": 81, "fps": 16},
}


def check_entry(d):
    name = os.path.basename(d.rstrip("/"))
    meta_p = os.path.join(d, "meta.json")
    if not os.path.exists(meta_p):
        return []
    meta = json.load(open(meta_p, encoding="utf-8"))
    model = meta.get("model")
    spec = SPECS.get(model)
    problems = []
    oe = meta.get("official_eval") or {}

    if meta.get("status") == "official-eval":
        if not oe:
            problems.append("status=official-eval 但 meta.json 缺 official_eval 块")
        elif oe.get("ok") != oe.get("total"):
            problems.append(f"official_eval 未全绿：{oe.get('ok')}/{oe.get('total')}")

        stats_p = os.path.join(d, "official-eval", "stats.json")
        if not os.path.exists(stats_p):
            problems.append("缺 official-eval/stats.json（无产物规格证据）")
        elif spec:
            st = json.load(open(stats_p, encoding="utf-8"))
            viol = st.get("spec_violation_count")
            if viol:
                problems.append(f"stats.json 记录 {viol} 条规格违规（{str(st.get('violations'))[:120]}）")
            if spec.get("kind") == "video":
                ds = st.get("dur_s") or {}
                expect = spec["frames"] / spec["fps"]
                for k in ("min", "max"):
                    v = ds.get(k)
                    if v is None:
                        problems.append(f"stats.json 缺 dur_s.{k}")
                    elif abs(v - expect) > 0.25:
                        problems.append(f"时长 {v}s 与 {spec['frames']}帧/{spec['fps']}fps 期望 {expect}s 不符")
            if spec.get("kind") == "audio":
                # 兼容两种落盘格式：AuK/Wan 风格 {sr, ch} 与 Music3 风格 {sample_rate, channels}
                for keys, want in ((("sr", "sample_rate"), spec["sample_rate"]),
                                   (("ch", "channels"), spec["channels"])):
                    got = None
                    for k in keys:
                        v = st.get(k)
                        if v is None:
                            continue
                        got = v[0] if isinstance(v, list) and v else v
                        break
                    if got is not None and want is not None and got != want:
                        problems.append(f"音频 {keys[0]}={got} 与期望 {want} 不符")
    return [(name, p) for p in problems]


def main():
    targets = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "models", "*")))
    bad = []
    for d in targets:
        if os.path.isdir(d):
            bad += check_entry(d)
    if bad:
        print("[products] 发现不一致：")
        for name, p in bad:
            print(f"  ✗ {name}: {p}")
        return 1
    print(f"[products] 检查 {len(targets)} 个目录，全部一致 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
