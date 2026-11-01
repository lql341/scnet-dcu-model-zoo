#!/usr/bin/env python3
# 正确性证据：按官方基线文件写死的配置调用服务跑固定子集，和官方 GPU 基线比；只罚劣化，不罚提升。
#
# ⚠️ 本件当前状态：baselines/<模型>.json 与 data/ 为官方下发物，尚未获取（见 ../MISSING.md）。
#    在官方文件到位前，本脚本会以 "缺少评测器/基线文件" 主动退出，不会伪造任何分数。
import os, re, sys, json, argparse, urllib.request
from concurrent.futures import ThreadPoolExecutor

TOL = 0.02  # 统一容差：劣化不超过 2%；有满分的按满分计（百分制即 2 pt），否则按基线值计
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 访问本机服务，不走代理


def check(got, m):
    # 返回明细；越低越好的指标（WER、AAS 等）在基线文件里标 lower_is_better
    worse = got - m["ref"] if m.get("lower_is_better") else m["ref"] - got
    tol = TOL * m.get("scale", m["ref"])
    return {"got": round(got, 2), "ref": m["ref"], "worse": round(worse, 2),
            "tol": round(tol, 2), "pass": worse <= tol + 1e-9}


def chat(prompt, max_tokens, extra):
    body = {"model": os.environ["SERVED_NAME"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": max_tokens, **extra}  # greedy；extra 为写死的思考开关等
    req = urllib.request.Request(
        f"http://127.0.0.1:{os.environ['PORT']}/v1/chat/completions",
        json.dumps(body).encode(), {"Content-Type": "application/json"})
    with OPENER.open(req, timeout=3600) as r:
        text = json.load(r)["choices"][0]["message"].get("content") or ""
    return text.split("</think>")[-1]  # 未开 reasoning parser 时去掉思考段


def gsm8k(item, max_tokens, extra):
    out = chat(item["question"] + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
               max_tokens, extra)
    found = re.findall(r"\\boxed\{([^{}]*)\}", out) or re.findall(r"-?\d[\d,]*(?:\.\d+)?", out)
    try:
        return bool(found) and float(found[-1].replace(",", "").strip(" $\\")) == float(item["answer"])
    except ValueError:
        return False


def mmlu(item, max_tokens, extra):
    options = "\n".join(f"{k}. {v}" for k, v in zip("ABCD", item["choices"]))
    out = chat(f"{item['question']}\n{options}\nAnswer with the option letter only.", max_tokens, extra)
    found = re.findall(r"\b[ABCD]\b", out)
    return bool(found) and found[0] == item["answer"]


EVALUATORS = {"gsm8k": gsm8k, "mmlu": mmlu}  # T02–T04 的评测器随分题材料下发，接口相同


def evaluate(name, m, extra):
    if name not in EVALUATORS:
        sys.exit(f"缺少评测器 {name}，请从分题材料获取")
    with open(m["subset"], encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    if len(items) != m["n"]:  # 子集题数写死，防止换子集
        sys.exit(f"{name} 子集应为 {m['n']} 题，实际 {len(items)} 题")
    with ThreadPoolExecutor(16) as ex:
        hits = list(ex.map(lambda it: EVALUATORS[name](it, m["max_tokens"], extra), items))
    return 100.0 * sum(hits) / len(items)


def selfcheck():
    # 逻辑自检：劣化超容差必须 FAIL，容差内与提升必须 PASS
    acc, wer = {"ref": 80.0, "scale": 100}, {"ref": 5.0, "scale": 100, "lower_is_better": True}
    assert not check(77.9, acc)["pass"], "掉 2.1 pt 必须 FAIL"
    assert check(78.0, acc)["pass"] and check(85.0, acc)["pass"], "掉 2 pt 以内或涨点必须 PASS"
    assert not check(7.1, wer)["pass"] and check(4.0, wer)["pass"], "WER 升超容差 FAIL，下降 PASS"
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    if ap.parse_args().selfcheck:
        selfcheck()
        sys.exit(0)

    # 官方基线文件：写死子集与题数、max_tokens、思考开关和 GPU 基线分数，参赛者不得修改
    base_path = f"baselines/{os.environ['MODEL_ID'].replace('/', '__')}.json"
    if not os.path.exists(base_path):
        sys.exit(f"[BLOCKED] 缺少官方基线文件 {base_path}\n"
                 f"          该文件由官方下发且不允许自行编写，详见 MISSING.md")
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    extra = base.get("extra_body", {})
    detail = {name: check(evaluate(name, m, extra), m) for name, m in base["metrics"].items()}
    ok = all(d["pass"] for d in detail.values())
    result = {k: os.environ.get(k) for k in ("MODEL_ID", "CARD", "FRAMEWORK", "PRECISION")}
    result.update(detail=detail, passed=ok)
    with open("verify_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if ok else 1)
