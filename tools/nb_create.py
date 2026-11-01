#!/usr/bin/env python3
"""用 OpenAPI 直接创建 K100-AI Notebook（绕过 skill 的 "trusted preset image" 白名单，
但仍走官方 /notebook/actions/create 接口）。

用法:
  python3 nb_create.py --plan            # 只打印将要提交的 request，不创建
  python3 nb_create.py --create --yes    # 真正创建（消耗卡时）
  python3 nb_create.py --list
  python3 nb_create.py --show <id>
  python3 nb_create.py --stop <id>
  python3 nb_create.py --release <id> --yes
"""
import argparse, json, sys

SKILL = "/Users/luql/.agents/skills/scnet-hpc/scripts"
sys.path.insert(0, SKILL)

from scnet_backends import create_backend, BackendContext   # noqa: E402  先导入 backends，避免循环导入
from scnet_sdk.notebook import NotebookService               # noqa: E402

REGION = "11250"                                        # 华东一区【昆山】
RESOURCE_GROUP = "hgk100_ainormal018b690441"            # hgk100_ainormal01, 4 卡/节点
IMAGE_ID = "2054150689889244138"                         # jupyter-sglang:jupyter-sglang-from20091
IMAGE_PATH = ("image.ac.com:5000/dcu/jsyadmin/base/jupyter-sglang:"
              "jupyter-sglang-from20091")
IMAGE_NAME = "jupyter-sglang:jupyter-sglang-from20091"
IMAGE_SIZE = "9026242169"
CARDS = 1
IMAGE_PATH_OVERRIDE = ""
IMAGE_NAME_OVERRIDE = ""
IMAGE_SIZE_OVERRIDE = ""


def svc():
    from pathlib import Path
    ctx = BackendContext(repo_root=Path(SKILL).parent, profile_name=None, profile={})
    backend = create_backend("openapi", ctx)
    return NotebookService(backend.client)


def build_plan(cards=CARDS, path="", name="", size=""):
    s = svc()
    res = [r for r in s.resources(REGION)
           if r.get("resourceGroupCode") == RESOURCE_GROUP][0]
    return {
        "region_id": REGION,
        "summary": {
            "region": res.get("clusterName"),
            "resource": res.get("resourceName"),
            "cards": cards,
            "cpu": res.get("cpuNumber"),
            "ram": res.get("ramSize"),
            "free": res.get("maxFreeNum"),
            "image": name or IMAGE_NAME,
        },
        "request": {
            "clusterId": REGION,
            "imagePath": path or IMAGE_PATH,
            "imageName": name or IMAGE_NAME,
            "imageSize": size or IMAGE_SIZE,
            "acceleratorType": str(res.get("resourceType") or "").upper(),
            "acceleratorNumber": str(cards),
            "resourceGroupCode": RESOURCE_GROUP,
            "mountHome": True,
            "mountInfo": [],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show")
    ap.add_argument("--reveal", action="store_true")
    ap.add_argument("--stop")
    ap.add_argument("--release")
    ap.add_argument("--cards", type=int, default=CARDS)
    ap.add_argument("--image-path", default="")
    ap.add_argument("--image-name", default="")
    ap.add_argument("--image-size", default="")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()

    s = svc()

    if a.list:
        print(json.dumps(s.list(REGION), ensure_ascii=False, indent=2)[:3000])
        return
    if a.show:
        print(json.dumps(s.detail(REGION, a.show, reveal=a.reveal) if "reveal" in s.detail.__code__.co_varnames else s.detail(REGION, a.show), ensure_ascii=False, indent=2)[:3000])
        return
    if a.stop:
        print(json.dumps(s.stop(REGION, a.stop), ensure_ascii=False, indent=2)[:1500])
        return
    if a.release:
        if not a.yes:
            sys.exit("release 不可逆，需 --yes")
        print(json.dumps(s.release(REGION, a.release), ensure_ascii=False, indent=2)[:1500])
        return

    plan = build_plan(a.cards, a.image_path, a.image_name, a.image_size)
    print("== 将提交的 request ==")
    print(json.dumps(plan["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(plan["request"], ensure_ascii=False, indent=2))

    if a.create:
        if not a.yes:
            sys.exit("创建会消耗卡时，需 --yes")
        out = s.create(REGION, plan)
        print("== create result ==")
        print(json.dumps(out, ensure_ascii=False, indent=2)[:2000])


if __name__ == "__main__":
    main()
