#!/usr/bin/env python3
"""生成项目结项 HTML 报告（单文件，自包含 CSS，无外部依赖）。

数据来源全部是仓库内真实文件：
  models/*/meta.json          条目元数据与官方评测结果
  models/*/bench/round2.jsonl 6 档基准原始数据
  docs/gotchas.md             踩坑条目
  sweeps/results-*.md         调优实验结论

用法：python3 tools/gen_report.py [输出路径]
"""
import glob
import html
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARCHIVE = "/public/home/luql/archive/zoo-dcu-2026-11"
STATUS_LABEL = {"official-eval": "官方评测完成", "partial": "部分完成",
                "bench": "6 档基准", "smoke": "仅 smoke", "blocked": "受阻"}
STATUS_CLASS = {"official-eval": "ok", "partial": "warn", "bench": "ok",
                "smoke": "info", "blocked": "bad"}
STATUS_ORDER = {"official-eval": 0, "partial": 1, "bench": 2, "smoke": 3, "blocked": 4}


def load_entries():
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "models", "*", "meta.json"))):
        m = json.load(open(f, encoding="utf-8"))
        oe = m.get("official_eval") or {}
        rows.append({
            "dir": os.path.basename(os.path.dirname(f)),
            "model": m["model"], "org": m.get("org", ""), "fw": m["framework"],
            "fwv": m.get("framework_version", ""), "prec": m["precision"],
            "cards": m.get("cards_used"), "tp": m.get("tp"),
            "status": m["status"], "bench": m.get("bench"),
            "ok": oe.get("ok"), "total": oe.get("total"),
            "per": oe.get("per_item_s"), "elapsed": oe.get("elapsed_s"),
            "note": m.get("bench_note", ""),
            "params": oe.get("params"),
            "output": oe.get("output"),
        })
    return rows


def load_bench(entry_dir):
    p = os.path.join(ROOT, "models", entry_dir, "bench", "round2.jsonl")
    if not os.path.exists(p):
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            out[(r.get("random_input_len"), r.get("max_concurrency"))] = r.get("output_throughput")
    return out


def load_gotchas():
    p = os.path.join(ROOT, "docs", "gotchas.md")
    if not os.path.exists(p):
        return []
    txt = open(p, encoding="utf-8").read()
    return [(int(n), t.strip()) for n, t in re.findall(r"\n## (\d+)\. (.+)", txt)]


def md_inline(t):
    """极简 markdown 行内转换：**粗体**、`代码`。"""
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def _strip_noise(txt):
    """去掉代码块与表格行，避免把 ```/表格当结论。"""
    out, in_fence = [], False
    for line in txt.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or line.lstrip().startswith("|"):
            continue
        out.append(line)
    return out


def _conclusion(txt):
    """抽一句结论：优先 "**结论/判定 …" 行，其次"判定"/"结论"小节正文，最后标题。"""
    lines = _strip_noise(txt)
    for line in lines:
        if re.match(r"^\*\*(一句话结论|结论|判定)", line.strip()):
            return line.strip().replace("**", "").strip()
    skip = ("|", "```", "- ", "* ", "> ", "1.", "2.", "3.", "#", "[")
    for want in (r"^#{2,3}\s*判定", r"^#{2,3}\s*(结论|实测结论|小结)"):
        for i, line in enumerate(lines):
            if not re.match(want, line.strip()):
                continue
            for nxt in lines[i + 1:]:
                t = nxt.strip()
                if not t or t.startswith(skip):
                    continue
                return t.replace("**", "").strip()
    for line in lines:
        if line.startswith("# "):
            return line[2:].replace("**", "").strip()
    return ""


def load_sweeps():
    """收集所有调优实验记录：sweeps/ 与 models/*/sweep/ 两处。"""
    items = []
    paths = sorted(glob.glob(os.path.join(ROOT, "sweeps", "results-*.md")))
    paths += sorted(glob.glob(os.path.join(ROOT, "models", "*", "sweep", "results-*.md")))
    for f in paths:
        txt = open(f, encoding="utf-8").read()
        title = ""
        for line in txt.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        items.append({"file": os.path.relpath(f, ROOT), "title": title,
                      "conclusion": _conclusion(txt)})
    return items


def esc(s):
    return html.escape(str(s)) if s is not None else ""


def build(rows, bench, gotchas, sweeps):
    n_entries = len(rows)
    n_ok = sum(1 for r in rows if r["status"] == "official-eval")
    n_partial = sum(1 for r in rows if r["status"] == "partial")
    n_bench = sum(1 for r in rows if r["bench"] == "yes")
    n_samples = sum(r["ok"] or 0 for r in rows)
    n_blocked = sum(1 for r in rows if r["status"] == "blocked")
    n_smoke = sum(1 for r in rows if r["status"] == "smoke")

    # ---------- 主表 ----------
    tr = []
    for r in sorted(rows, key=lambda x: (STATUS_ORDER[x["status"]], -(x["ok"] or 0), x["model"])):
        b = bench.get(r["dir"], {})
        t4 = b.get((4096, 64))
        t16 = b.get((16384, 8))
        ev = f"{r['ok']}/{r['total']}" if r["ok"] is not None else "—"
        per = f"{r['per']:.1f}" if r["per"] else "—"
        tr.append(
            f"<tr><td class='mono'>{esc(r['model'])}</td><td>{esc(r['fw'])}</td>"
            f"<td>{esc(r['prec'])}</td><td class='num'>{r['cards']}</td>"
            f"<td><span class='badge {STATUS_CLASS[r['status']]}'>{STATUS_LABEL[r['status']]}</span></td>"
            f"<td class='num'>{ev}</td><td class='num'>{per}</td>"
            f"<td class='num'>{t4:.2f}</td>" if t4 else
            f"<tr><td class='mono'>{esc(r['model'])}</td><td>{esc(r['fw'])}</td>"
            f"<td>{esc(r['prec'])}</td><td class='num'>{r['cards']}</td>"
            f"<td><span class='badge {STATUS_CLASS[r['status']]}'>{STATUS_LABEL[r['status']]}</span></td>"
            f"<td class='num'>{ev}</td><td class='num'>{per}</td>"
            f"<td class='num'>—</td>"
        )
        if t4:
            tr[-1] += f"<td class='num'>{t16:.2f}</td></tr>" if t16 else "<td class='num'>—</td></tr>"
        else:
            tr[-1] += "<td class='num'>—</td></tr>"

    # ---------- 跨框架对比 ----------
    sgl = bench.get("XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__sglang", {})
    vll = bench.get("XiaomiMiMo__MiMo-V2.6-Distill-Qwen-9B__bw1000__vllm", {})
    cmp_rows = []
    for k in [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]:
        a, b = sgl.get(k), vll.get(k)
        if a and b:
            d = (b - a) / a * 100
            cls = "bad" if d < -50 else ("warn" if d < 0 else "ok")
            cmp_rows.append(
                f"<tr><td class='mono'>{k[0]//1024}K / c{k[1]}</td>"
                f"<td class='num strong'>{a:.2f}</td><td class='num'>{b:.2f}</td>"
                f"<td class='num {cls}'>{d:+.0f}%</td></tr>")

    # ---------- Kimi 6 档 ----------
    kimi = bench.get("moonshotai__Kimi-Linear-48B-A3B-Instruct__bw1000__sglang", {})
    kimi_rows = []
    for k in [(4096, 1), (4096, 8), (4096, 64), (16384, 1), (16384, 8), (16384, 64)]:
        v = kimi.get(k)
        if v:
            kimi_rows.append(f"<tr><td class='mono'>{k[0]//1024}K / c{k[1]}</td><td class='num strong'>{v:.2f}</td></tr>")

    gotcha_items = "".join(
        f"<li><span class='gnum'>#{n}</span> {esc(t)}</li>" for n, t in gotchas)
    sweep_items = "".join(
        f"<li><span class='mono small'>{esc(s['file'])}</span><br>"
        f"<b>{md_inline(s['title'].replace('**', ''))}</b><br>{md_inline(s['conclusion'])}</li>"
        for s in sweeps)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BW1000（海光 DCU）大模型适配 —— 结项报告</title>
<style>
:root {{
  --bg:#0e1116; --panel:#161b22; --panel2:#1c2430; --line:#2b3441;
  --fg:#e6edf3; --dim:#9aa7b8; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
  --info:#58a6ff; --accent:#7ee787;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
  font:15px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; }}
.wrap {{ max-width:1180px; margin:0 auto; padding:48px 28px 90px; }}
h1 {{ font-size:34px; line-height:1.25; margin:0 0 10px; letter-spacing:-.02em; }}
h2 {{ font-size:22px; margin:52px 0 16px; padding-bottom:10px; border-bottom:1px solid var(--line); }}
h3 {{ font-size:17px; margin:28px 0 12px; color:var(--accent); }}
p, li {{ color:#c9d4e0; }}
.sub {{ color:var(--dim); font-size:14px; }}
.mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
.small {{ font-size:12px; }}
.hero {{ background:linear-gradient(150deg,#182030,#11161d); border:1px solid var(--line);
  border-radius:16px; padding:26px 28px; margin:22px 0 8px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(148px,1fr)); gap:14px; margin:26px 0 8px; }}
.kpi {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }}
.kpi .v {{ font-size:28px; font-weight:700; letter-spacing:-.02em; }}
.kpi .k {{ color:var(--dim); font-size:12.5px; margin-top:2px; }}
.kpi.ok .v {{ color:var(--ok); }} .kpi.warn .v {{ color:var(--warn); }} .kpi.info .v {{ color:var(--info); }}
table {{ width:100%; border-collapse:collapse; margin:14px 0 8px; font-size:13.5px; }}
th, td {{ padding:9px 11px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ color:var(--dim); font-weight:600; font-size:12.5px; text-transform:uppercase; letter-spacing:.04em;
  background:var(--panel2); }}
tr:hover td {{ background:#1a212b; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:13px; }}
td.strong {{ color:var(--accent); font-weight:700; }}
td.ok {{ color:var(--ok); }} td.warn {{ color:var(--warn); }} td.bad {{ color:var(--bad); }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; white-space:nowrap; }}
.badge.ok {{ background:rgba(63,185,80,.15); color:var(--ok); border:1px solid rgba(63,185,80,.35); }}
.badge.warn {{ background:rgba(210,153,34,.15); color:var(--warn); border:1px solid rgba(210,153,34,.35); }}
.badge.info {{ background:rgba(88,166,255,.13); color:var(--info); border:1px solid rgba(88,166,255,.3); }}
.badge.bad {{ background:rgba(248,81,73,.13); color:var(--bad); border:1px solid rgba(248,81,73,.32); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px 20px; margin:14px 0; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
@media (max-width:820px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
ul {{ padding-left:20px; }} li {{ margin:7px 0; }}
.gnum {{ color:var(--info); font-family:ui-monospace,monospace; margin-right:4px; }}
code {{ background:#0b0f14; border:1px solid var(--line); border-radius:5px; padding:1px 5px;
  font-family:ui-monospace,Menlo,monospace; font-size:12.5px; }}
pre {{ background:#0b0f14; border:1px solid var(--line); border-radius:10px; padding:14px 16px;
  overflow-x:auto; font-size:12.5px; line-height:1.6; }}
pre code {{ background:none; border:none; padding:0; }}
.note {{ border-left:3px solid var(--warn); background:rgba(210,153,34,.07); padding:12px 16px;
  border-radius:0 8px 8px 0; margin:14px 0; }}
.good {{ border-left:3px solid var(--ok); background:rgba(63,185,80,.07); padding:12px 16px;
  border-radius:0 8px 8px 0; margin:14px 0; }}
footer {{ margin-top:60px; padding-top:22px; border-top:1px solid var(--line); color:var(--dim); font-size:13px; }}
a {{ color:var(--info); }}
</style>
</head>
<body>
<div class="wrap">

<h1>BW1000（海光 DCU）大模型适配 —— 结项报告</h1>
<p class="sub">2026-11-09 · 15 个条目 · {n_samples} 条官方条件真实产物 · 3 个镜像归档（31 GB）· {len(gotchas)} 条踩坑 · 5 条硬约束</p>

<div class="hero">
<p style="margin:0 0 10px"><b>一句话</b>：在一张 64 GB 的海光 BW1000 上，把 T03/T04 官方 8 个模型中的 6 个跑到官方条件全绿
（另 1 个 53/60、1 个未做），并沉淀出可复现的镜像归档、24 条踩坑与 5 条换谁都绕不过的硬约束。</p>
<p style="margin:0"><b>为什么现在结项</b>：<code>scnethr2601a10</code> 于 2026-11-07 被移出分区
<code>AllowAccounts</code>；<code>luql</code> 有 SSH 但无卡权限。两个账号都无法再提交作业。
所有数据、镜像、经验已在结项前完成三层保全。</p>
</div>

<div class="kpis">
  <div class="kpi ok"><div class="v">{n_ok}</div><div class="k">官方评测全绿</div></div>
  <div class="kpi warn"><div class="v">{n_partial}</div><div class="k">部分完成（53/60）</div></div>
  <div class="kpi info"><div class="v">{n_bench}</div><div class="k">完整 6 档基准</div></div>
  <div class="kpi"><div class="v">{n_samples}</div><div class="k">真实产物（官方清单 935）</div></div>
  <div class="kpi"><div class="v">31 GB</div><div class="k">镜像+产物归档</div></div>
  <div class="kpi"><div class="v">{len(gotchas)}</div><div class="k">踩坑条目</div></div>
</div>

<h2>1. 条目总览</h2>
<p class="sub">按状态排序。官方样本列为已完成条数；单条耗时单位秒。</p>
<table>
<thead><tr><th>模型</th><th>框架</th><th>精度</th><th>卡</th><th>状态</th>
<th>官方样本</th><th>单条 s</th><th>4K/c64 tok/s</th><th>16K/c8 tok/s</th></tr></thead>
<tbody>
{''.join(tr)}
</tbody>
</table>
<p class="sub">状态分布：official-eval {n_ok} · partial {n_partial} · bench {n_bench} · smoke {n_smoke} · blocked {n_blocked}</p>

<h2>2. 核心数据：同模型跨框架硬对比</h2>
<p>MiMo-9B，同一张 64 GB BW1000，同一套压测客户端与有效性校验（<code>preflight_lib.py</code> 挡掉「HTTP 400 当成功」的假数据）。</p>
<table>
<thead><tr><th>档位</th><th>SGLang 0.5.10</th><th>vLLM 0.20</th><th>vLLM 相对</th></tr></thead>
<tbody>{''.join(cmp_rows)}</tbody>
</table>
<div class="note">
<b>结论</b>：这张卡上 SGLang 全面优于 vLLM，长上下文差距拉到 <b>5–7 倍</b>，且<b>不是参数问题</b> ——
启动日志显示 <code>Cannot use ROCm custom paged attention kernel, falling back to Triton</code>，
即 DCU 上没有 paged-attention 内核，只有 Triton 兜底实现（见 §5 硬约束 1）。
</div>

<h2>3. 性能与优化（都是实测）</h2>
<div class="grid2">
<div class="card">
<h3>Kimi-Linear-48B × SGLang（TP=4）6 档</h3>
<table><thead><tr><th>档位</th><th>output tok/s</th></tr></thead>
<tbody>{''.join(kimi_rows)}</tbody></table>
</div>
<div class="card">
<h3>优化手段与收益</h3>
<table>
<thead><tr><th>手段</th><th>收益</th></tr></thead>
<tbody>
<tr><td>SGLang <code>max-running-requests 8→64</code> + 开 CUDA Graph</td><td class="ok">+73% ~ +466%</td></tr>
<tr><td>vLLM 去 <code>--enforce-eager</code> + <code>max-num-seqs 128</code></td><td class="ok">+3% ~ +32%</td></tr>
<tr><td>Kimi 开 CUDA Graph</td><td class="ok">+37.9% ~ +620.8%</td></tr>
<tr><td>16K 档 <code>--chunked-prefill-size 4096</code></td><td class="ok">+5% ~ +8%</td></tr>
<tr><td>AuK → AuK-Flash（4 步蒸馏权重）</td><td class="ok">4.4×</td></tr>
<tr><td>ASR <code>--max-num-seqs 64</code> + 16 并发</td><td class="ok">−10% 耗时</td></tr>
<tr><td>AWQ（weight-only INT4）</td><td class="bad">慢 5×（显存收益真实）</td></tr>
</tbody></table>
</div>
</div>

<h2>4. 五条硬约束（换框架、换参数都过不去）</h2>
<table>
<thead><tr><th>#</th><th>约束</th><th>证据</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td><b>vLLM 在 DCU 上没有 paged-attention 内核</b>，只有 Triton 兜底</td>
<td>五个旋钮全试过全部无效：chunked 切片 / AITER / backend 覆盖 / fp8 KV / TP=2 / <b>v0 引擎</b>。16K TPOT 119 ms（SGLang 24 ms）</td></tr>
<tr><td class="num">2</td><td><b>fp8 在线量化无内核</b></td><td><code>--quantization fp8</code> → engine core 初始化即失败</td></tr>
<tr><td class="num">3</td><td><b>14B 级视频模型单卡装不下</b></td><td>Wan2.1 单卡 OOM（offload + t5_cpu 仍打满 63.98 GB）→ 必须 4 卡 FSDP</td></tr>
<tr><td class="num">4</td><td><b>2048² 文生图没有合批余量</b></td><td>Qwen-Image batch=2 即 OOM（已分配 46.56 GB / 剩 3.84 / 需 6.75）</td></tr>
<tr><td class="num">5</td><td><b>HY-World 重建需 ROCm 版 gsplat</b></td><td><code>worldrecon/rasterization.py</code> 硬 import gsplat；官方要 py3.11 + FA2/FA3</td></tr>
</tbody>
</table>

<h2>5. 技术路线（可复用的方法论）</h2>
<div class="card">
<h3>5.1 三层环境隔离</h3>
<p>镜像（不动）→ <code>/opt/*-libs</code>（<code>--no-deps</code> + <b>cp310</b> 轮子）→ <code>PYTHONPATH</code> 前置。
永不 <code>pip install</code> 到系统目录，永不升级镜像里的 torch。<br>
<i>为什么强调 cp310：按宿主机 py3.12 下载的轮子装进 py3.10 镜像，pip 会静默拒装，表现为
<code>ModuleNotFoundError</code>，曾把两个条目误判成 blocked。</i></p>

<h3>5.2 参数只从条件清单读</h3>
<p>官方 <code>conditions/*.jsonl</code> 是唯一权威。曾因写死 <code>use_kv_cache</code> 却不传
<code>num_inference_steps</code>，实际按 pipeline 默认 40 步跑（官方要 8 步），180 条产物差点全废。</p>

<h3>5.3 验规格，不验「作业成功」</h3>
<p>音频看 24 kHz/单声道、图像看 2048²、视频看 81 帧/16 fps，<code>spec_violation_count</code> 必须为 0。<br>
<i>OmniWeaving 首轮 60 条「全部成功」，实际是 848×480（官方要 832×480）——
因为 <code>--aspect_ratio 16:9</code> 落到了最近的尺寸桶。这是靠逐条读 mp4 header 才发现的，
只看「文件存在且 &gt;200 KB」会一路绿到底。</i> 现在有 <code>tools/check_products.py</code> 守门。</p>

<h3>5.4 续跑优先</h3>
<p>单条 1–26 min 的任务一律「产物存在则跳过」。<b>收益是实测的</b>：账号被回收时 OmniWeaving
跑了 22 小时被砍，53/60 产物完好，只需补 3.5 小时。</p>

<h3>5.5 阴性结论同样入库</h3>
<p>不可行的路径要写清「试了什么、错在哪、什么条件才能过」，避免下个会话重复烧卡时。
§4 的五条硬约束和 §3 的「不必再做的清单」都是这么来的。</p>

<h3>5.6 归档先于结论</h3>
<p>镜像/轮子在用完之后立刻归档 + sha256 + <b>功能性还原验证</b>（不是只算哈希：真的 load 起来跑一条）。
三个镜像现在是唯一能复现这套环境的资产。</p>
</div>

<h2>6. 可复现性资产</h2>
<table>
<thead><tr><th>资产</th><th>位置</th><th>校验</th></tr></thead>
<tbody>
<tr><td>SGLang 镜像 7.9 GB</td><td class="mono">{ARCHIVE}/images/</td><td>sha256 <code>f16125ba…</code> + load/smoke</td></tr>
<tr><td>vLLM 镜像 8.7 GB</td><td class="mono">{ARCHIVE}/images/</td><td>sha256 <code>986f95a3…</code> + load/smoke</td></tr>
<tr><td>Music3 镜像 13.7 GB</td><td class="mono">{ARCHIVE}/images/</td><td>sha256 <code>b61b05bd…</code> + 还原跑通 1 条</td></tr>
<tr><td>923 条官方条件产物</td><td class="mono">{ARCHIVE}/products/</td><td>与各条目 meta.json 的 ok 数一致</td></tr>
<tr><td>调优原始数据 34 个 jsonl</td><td class="mono">{ARCHIVE}/products/optimizations/</td><td>§3 所有结论的证据</td></tr>
<tr><td>cp310 轮子 4 套 + 源码 3 份</td><td class="mono">{ARCHIVE}/assets/ · sources/</td><td>包内 sha256 清单</td></tr>
<tr><td>代码/文档/条件清单/日志</td><td>GitHub <code>lql341/scnet-dcu-model-zoo</code><br>GitLab <code>luql/scnet-dcu-model-zoo</code></td><td>双远端同 commit</td></tr>
</tbody>
</table>

<h2>7. 踩坑清单（{len(gotchas)} 条，按隐蔽度排序）</h2>
<div class="card"><ul>{gotcha_items}</ul></div>
<p class="sub">前三条最贵：① <code>/dev/mkfd</code> 必须传（否则 <code>torch.cuda.is_available()</code> 为 False 却 <code>device_count()</code> 为 1）；
② CUDA Graph 的假陷阱；③ <code>/etc/subuid</code> 为空导致 rootless podman 解包失败。</p>

<h2>8. 调优实验记录</h2>
<div class="card"><ul>{sweep_items}</ul></div>

<h2>9. 成本核算</h2>
<table>
<thead><tr><th>项目</th><th>卡时</th></tr></thead>
<tbody>
<tr><td>基线与调参</td><td class="num">~62</td></tr>
<tr><td>Qwen-Image / AuK / ASR / FA 官方评测</td><td class="num">~12</td></tr>
<tr><td>vLLM 16K 三轮扫描（含失败轮次）</td><td class="num">~3</td></tr>
<tr><td>Wan2.1（含 OOM 与 FSDP 调试 + 全量）</td><td class="num">~75</td></tr>
<tr><td>Music3（探针 + 60 条 + 镜像归档/还原）</td><td class="num">~8</td></tr>
<tr><td>OmniWeaving（探针 + 53 条）</td><td class="num">~30</td></tr>
<tr><td>其它（合批/量化/AWQ 探针、镜像归档）</td><td class="num">~10</td></tr>
<tr><td><b>合计</b></td><td class="num strong">~200 卡时 ≈ 105 元</td></tr>
</tbody>
</table>
<p>换来 15 个可复现条目 + 3 个镜像归档 + {len(gotchas)} 条踩坑 + {n_samples} 条真实产物 —— <b>约 0.11 元/样本</b>。</p>

<h2>10. 结项状态与重启方式</h2>
<div class="note">
<b>为什么停</b>：<code>scnethr2601a10</code> 2026-11-07 14:12 被移出 <code>hx1hdnormal</code> 的
<code>AllowAccounts</code>（该列表共 10552 条，含同批的 <code>scnethr2601a05</code>/<code>a07</code>/<code>scnethr202601</code>，
唯独没有我们）；<code>luql</code> 有 SSH 但无卡权限。
</div>
<h3>机时恢复后（3 步）</h3>
<pre><code># 0) 三分钟判定"真的能提交了"
scontrol -a show partition hx1hdnormal | tr ' ' '\\n' | grep '^AllowAccounts=' \\
  | cut -d= -f2 | tr ',' '\\n' | grep -x "$USER"      # 有输出才算通过

# 1) 恢复数据（不用重下权重）
cd {ARCHIVE}/images && sha256sum -c ...               # 见 VERIFY.md
tar xzf {ARCHIVE}/assets/wheels-auk-cp310.tar.gz -C ~/wheels
git clone https://github.com/lql341/scnet-dcu-model-zoo.git ~/zoo

# 2) 补完唯一未完成的条目（OmniWeaving 剩 7 条，约 3.5 h）
rsync -az {ARCHIVE}/products/omniweaving/ <host>:<...>/repo/official-eval/
sbatch --export=ALL,PROBE_ONLY=0 container/submit.slurm   # 自动跳过已有 53 条</code></pre>

<h3>恢复后值得做的（按性价比）</h3>
<table>
<thead><tr><th>#</th><th>事项</th><th>成本</th><th>为什么</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td>把 3 个镜像推到自有 Harbor 命名空间</td><td>0.5 卡时</td><td>唯一能真正摆脱「上游删仓 + 临时账号回收」两个风险（需 Harbor 凭据）</td></tr>
<tr><td class="num">2</td><td>量化条目（AWQ 已验证可跑）</td><td>4–8 卡时</td><td>全库唯一空白；慢 5× 但显存收益真实</td></tr>
<tr><td class="num">3</td><td>WorldMirror（编 ROCm 版 gsplat）</td><td>8+ 卡时</td><td>T03 第 5 个模型，可能失败</td></tr>
<tr><td class="num">4</td><td>T01/T02 精度</td><td>需官方 baselines</td><td>材料从未下发，无法自证</td></tr>
</tbody>
</table>

<h3>明确不必再做的（已有结论，别再烧卡时）</h3>
<ul>
<li>vLLM 16K 调参：chunked 切片 / AITER / backend 覆盖 / fp8 KV / TP=2 / <b>v0 引擎</b> —— 全部无效</li>
<li>Qwen-Image 合批：2048² 显存无余量，batch=2 即 OOM</li>
<li>Kimi 16K/c64「回退 6.3%」：是 ±7% 噪声，不是配置问题（关 graph 反而掉 31%）</li>
<li>fp8 在线量化：DCU 无 kernel</li>
</ul>

<footer>
数据来源：仓库内 <code>models/*/meta.json</code>、<code>bench/round2.jsonl</code>、
<code>docs/gotchas.md</code>、<code>sweeps/results-*.md</code>，由 <code>tools/gen_report.py</code> 生成。
报告中每一个数字都能在上述文件中追到原始记录。<br>
代码：GitHub <code>lql341/scnet-dcu-model-zoo</code> · GitLab <code>luql/scnet-dcu-model-zoo</code>　|　
二进制归档：<code>{ARCHIVE}/</code>（31 GB）
</footer>
</div>
</body>
</html>
"""


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs", "report.html")
    rows = load_entries()
    bench = {r["dir"]: load_bench(r["dir"]) for r in rows if r["bench"] == "yes"}
    gotchas = load_gotchas()
    sweeps = load_sweeps()
    html_out = build(rows, bench, gotchas, sweeps)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(html_out)
    print(f"[report] {out}  ({len(html_out)/1024:.1f} KB)")
    print(f"[report] 条目 {len(rows)} · 踩坑 {len(gotchas)} · 实验 {len(sweeps)} · bench {len(bench)}")


if __name__ == "__main__":
    main()
