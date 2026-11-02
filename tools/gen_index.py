#!/usr/bin/env python3
"""从各条目的 meta.json 生成 INDEX.md 与 README.md 总矩阵。

单一数据源 = models/*/meta.json。改数字只改 meta.json，文档不再手写。
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP_KEYS = ['4k_c1', '4k_c8', '4k_c64', '16k_c1', '16k_c8', '16k_c64']


def load():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, 'models', '*/'))):
        m = json.load(open(os.path.join(d, 'meta.json'), encoding='utf-8'))
        m['_dir'] = os.path.basename(d.rstrip('/'))
        out.append(m)
    return out


def fmt(v):
    return '—' if v is None else f'{v:.2f}'


def main():
    ents = load()
    done = [e for e in ents if e['bench'] == 'yes']
    part = [e for e in ents if e['bench'] != 'yes' and e.get('status') != 'blocked']
    blocked = [e for e in ents if e.get('status') == 'blocked']

    # ---------- INDEX.md ----------
    lines = ['# 实测索引', '',
             '> 本文件由 `tools/gen_index.py` 从各条目 `meta.json` 生成，**请勿手改**。', '',
             '## 已完成 6 档基准', '',
             '| 模型 | 卡型×卡数 | 框架 | 精度 | TP | 冷启动 | 4K/c1 | 4K/c8 | 4K/c64 | 16K/c1 | 16K/c8 | 16K/c64 | 详情 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for e in done:
        cs = e['cold_start_s']
        lines.append(
            f"| {e['model']} | {e['card']} ×{e['cards_used']} | {e['framework']} {e['framework_version'].split('+')[0]} "
            f"| {e['precision']} | {e['tp']} | {cs[0]}–{cs[1]} s "
            + ''.join(f"| {fmt(e['throughput'].get(k))} " for k in TP_KEYS)
            + f"| [→](models/{e['_dir']}/) |")
    lines += ['', '单位：output tok/s（取 `bench/round2.jsonl`）。', '']

    if part:
        lines += ['## 已跑通但无 6 档基准', '',
                  '| 模型 | 卡型×卡数 | 框架 | 精度 | 状态 | 冷启动 | 原因 | 详情 |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for e in part:
            cs = e['cold_start_s']
            lines.append(f"| {e['model']} | {e['card']} ×{e['cards_used']} | {e['framework']} "
                         f"| {e['precision']} | {e['status']} | {cs[0]}–{cs[1]} s "
                         f"| {e['bench_note'] or '—'} | [→](models/{e['_dir']}/) |")
        lines += ['']

    lines += ['## 失败记录（blocked）', '',
              '跑不通的配置也留档，写明原因与证据，不删。', '',
              '| 组合 | 结果 | 原因 | 详情 |',
              '| --- | --- | --- | --- |']
    for e in blocked:
        lines.append(f"| {e['model']} × {e['card']} × {e['framework']} "
                     f"| ❌ 服务无法启动 | {e['bench_note']} | [→](models/{e['_dir']}/) |")
    lines += ['',
              '| 组合 | 结果 | 原因 |',
              '| --- | --- | --- |',
              '| Kimi-Linear × BW1000 × vLLM | ❌ 框架不支持 | vLLM 0.20 内置模型无 `kimi_linear`（只有 kimi_k25 / kimi_vl / kimi_audio） | gotchas #2 旁注 |',
              '| Kimi-Linear × BW1000 × SGLang × CUDA Graph | 🟡 未完成 | TP=4 报 `HIP error: invalid device ordinal` | gotchas #14 |',
              '| MiMo × K100-AI × vLLM | 🟡 仅 smoke | Notebook 内跑通，6 档基准未跑 | 见 README 附注 |',
              '| MiMo × K100-AI × SGLang | ❌ 镜像缺陷 | `jupyter-sglang` 镜像 aiter 缺 `ops.triton.gemm` | gotchas #2 |',
              '']
    open(os.path.join(ROOT, 'INDEX.md'), 'w', encoding='utf-8').write('\n'.join(lines))
    print('INDEX.md 已生成', len(done), '条基准 +', len(part), '条部分验证')

    # ---------- README.md 总矩阵（替换 <!-- MATRIX --> 区块） ----------
    rp = os.path.join(ROOT, 'README.md')
    rd = open(rp, encoding='utf-8').read()
    mat = ['<!-- MATRIX:BEGIN （由 tools/gen_index.py 生成，勿手改） -->',
           '| 模型 | 参数/权重 | 卡型 | 框架 | 精度 | 状态 | 4K/c64 | 16K/c8 |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for e in ents:
        st = '✅ 6 档基准' if e['bench'] == 'yes' else ('✅ 跑通（bench 口径未下发）' if e['status'] == 'smoke' else e['status'])
        mat.append(f"| [{e['model']}](models/{e['_dir']}/) | {e['card']} ×{e['cards_used']} | {e['framework']} "
                   f"| {e['precision']} | {st} | {fmt(e['throughput'].get('4k_c64'))} "
                   f"| {fmt(e['throughput'].get('16k_c8'))} |")
    mat.append('<!-- MATRIX:END -->')
    block = '\n'.join(mat)
    if '<!-- MATRIX:BEGIN' in rd:
        rd = re.sub(r'<!-- MATRIX:BEGIN.*?<!-- MATRIX:END -->', block, rd, flags=re.S)
    else:
        rd = rd.replace('## 跑通矩阵', '## 跑通矩阵\n\n' + block)
    open(rp, 'w', encoding='utf-8').write(rd)
    print('README.md 总矩阵已更新')


if __name__ == '__main__':
    main()
