#!/usr/bin/env python3
"""在 SCNet Notebook 里执行命令：通过 Jupyter Server REST + kernel websocket。

用法:
  python3 nb_exec.py "命令"                 # 执行 shell 命令
  python3 nb_exec.py --py "python 代码"     # 直接执行 python
"""
import argparse, json, ssl, sys, time, urllib.request, uuid

import websocket

import os
NB = os.environ.get("NB_ID", "2116026334807842817")
HOST = f"n-{NB}.ksai.scnet.cn:58043"
BASE = f"https://{HOST}/jupyter-forward/{NB}"
TOKEN = os.environ.get("NB_TOKEN", f"sothisai_{NB}")


def api(path, method="GET", body=None):
    url = f"{BASE}{path}"
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}token={TOKEN}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        raw = r.read()
        return json.loads(raw) if raw else None


def run(code, timeout=900):
    k = api("/api/kernels", "POST", {"name": "python3"})
    kid = k["id"]
    ws_url = (f"wss://{HOST}/jupyter-forward/{NB}/api/kernels/{kid}/channels"
              f"?token={TOKEN}")
    ws = websocket.create_connection(ws_url, timeout=timeout,
                                     sslopt={"cert_reqs": ssl.CERT_NONE})
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "u", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": False,
                    "user_expressions": {}, "allow_stdin": False,
                    "stop_on_error": True},
        "channel": "shell",
    }))
    out = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            m = json.loads(ws.recv())
        except Exception:
            break
        t = m.get("msg_type") or m.get("header", {}).get("msg_type")
        if t == "stream":
            out.append(m["content"].get("text", ""))
        elif t == "execute_result":
            out.append(m["content"]["data"].get("text/plain", ""))
        elif t == "error":
            out.append("\n".join(m["content"].get("traceback", [])))
        elif t == "execute_reply":
            status = m["content"].get("status")
            out.append(f"\n[status={status}]")
            break
    ws.close()
    try:
        api(f"/api/kernels/{kid}", "DELETE")
    except Exception:
        pass
    return "".join(out)


def put(local, remote):
    """通过 Jupyter contents API 上传文件（remote 相对 Notebook 根目录，如 api/x.sh）。"""
    body = {"type": "file", "format": "text",
            "content": open(local, encoding="utf-8").read()}
    r = api("/api/contents/" + remote, "PUT", body)
    return r


def get(remote, local):
    """下载 Notebook 里的文件（通过 /api/contents 的 file 下载）。"""
    import urllib.parse
    url = f"{BASE}/api/contents/{urllib.parse.quote(remote)}?token={TOKEN}&format=text"
    ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        data = json.loads(r.read())
    open(local, "w", encoding="utf-8").write(data.get("content") or "")
    return len(data.get("content") or "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--py")
    ap.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"))
    ap.add_argument("--get", nargs=2, metavar=("REMOTE", "LOCAL"))
    a = ap.parse_args()
    if a.put:
        print(json.dumps(put(*a.put), ensure_ascii=False)[:300])
        sys.exit(0)
    if a.get:
        print("bytes:", get(*a.get))
        sys.exit(0)
    if a.py:
        code = a.py
    elif a.cmd:
        code = ("import subprocess\n"
                f"print(subprocess.run({a.cmd!r}, shell=True, text=True,"
                " capture_output=True).stdout)\n"
                f"print(subprocess.run({a.cmd!r}, shell=True, text=True,"
                " capture_output=True).stderr)")
    else:
        sys.exit("need cmd or --py")
    print(run(code))
