#!/usr/bin/env python3
# 精度验证：需要官方 baselines/<model>.json 与评测器。
# T03 官方明示「不进行统一评分」，且我们的 baselines/ 一直为空 → 主动 BLOCKED，不编造分数。
import os, sys

BASE = f"baselines/{os.environ.get('MODEL_ID', 'MiniMaxAI/MiniMax-Music3').replace('/', '__')}.json"


def main():
    if os.path.exists(BASE):
        sys.exit(f"[BLOCKED] 发现基线文件 {BASE}，但官方 T03 未提供音乐评测器；"
                 f"请先确认评测协议再实现，禁止自行编造指标")
    sys.exit(f"[BLOCKED] 缺少官方基线/评测器 {BASE}（详见 MISSING.md）\n"
             f"          官方 T03 说明：不进行统一评分和性能比较，故本条目只交原始产物与规格统计")


if __name__ == "__main__":
    main()
