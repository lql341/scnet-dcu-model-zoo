# data/ —— 官方下发评测子集，当前为空

本目录应存放 `gsm8k.jsonl` 与 `mmlu.jsonl`，题数须与 `baselines/<模型>.json` 的 `n` 完全一致
（`verify.py` 会校验题数，防止换子集）。

缺失影响与索取方式见仓库根目录 `MISSING.md` 第 ② 条。
