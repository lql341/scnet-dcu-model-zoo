"""AuK 单条推理探针：用官方 seedtts 的一条参考音 + 目标文本，验证链路。"""
import json, os, sys, glob

COND = "/work/repo/conditions/auk.jsonl"
ASSETS = "/assets"
item = json.loads(open(COND).readline())
print("条件 id:", item["id"], "| model:", item["model"])
print("输入字段:", list(item.keys()))
print("assets:", json.dumps(item.get("assets"), ensure_ascii=False)[:300])
print("params:", json.dumps(item.get("params"), ensure_ascii=False)[:200])

# 解析参考音路径
rel = item["assets"]["reference_audio"]["path"]
p = os.path.normpath(os.path.join(os.path.dirname(COND), rel))
if not os.path.exists(p):
    r2 = rel
    while r2.startswith("../"): r2 = r2[3:]
    p = os.path.normpath(os.path.join(ASSETS, r2))
print("参考音:", p, "| 存在:", os.path.exists(p))

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
try:
    from auk.infer.infer_cli import main as cli
    print("auk CLI 可导入")
except Exception as e:
    print("auk CLI 导入失败:", type(e).__name__, str(e)[:200])
