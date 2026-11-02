"""nagisa 的最小 stub —— 仅用于避开 dyNET 这个重 C++ 依赖。

依据：qwen_asr/inference/qwen3_forced_aligner.py 里 nagisa 只被 `tokenize_japanese()`
使用（唯一一处 `nagisa.tagging(text).words`）。本条目评测集是 30 条**英文**
（LibriSpeech test-clean），不触达日语分词路径。

若将来要处理日语，必须安装真实 nagisa（依赖 dyNET），本 stub 会直接抛错而不是静默出错。
"""
class _Tagging:
    def __init__(self):
        self.words = []


def tagging(text):
    raise NotImplementedError(
        "nagisa 被 stub 替换：本条目只评测英文。如需日语分词请安装真实 nagisa==0.2.11（依赖 dyNET）。"
    )
