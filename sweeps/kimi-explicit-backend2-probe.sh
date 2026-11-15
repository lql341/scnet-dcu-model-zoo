#!/bin/bash
# 0.20 里显式 --attention-backend 到底在哪儿被消费？（上一个探测证明 selector.py 里没有这个参数）
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/explicit2.md
{
echo '### 0. AttentionSelectorConfig 的字段（是否有 backend 字段）'
grep -rn -A30 'class AttentionSelectorConfig' "$SITE"/vllm/v1/attention/ 2>/dev/null | head -45
echo
echo '### 1. 全包搜 attention_backend 的消费点（只看 .py，排除 upstream 注释）'
grep -rn --include='*.py' 'attention_backend' "$SITE"/vllm/ 2>/dev/null \
  | grep -v 'test' | head -40
echo
echo '### 2. 全包搜 selected_backend'
grep -rn --include='*.py' 'selected_backend' "$SITE"/vllm/ 2>/dev/null | head -30
echo
echo '### 3. platforms/__init__.py 的 get_attn_backend_cls'
grep -rn -A25 'def get_attn_backend_cls' "$SITE"/vllm/platforms/*.py 2>/dev/null | head -60
echo
echo '### 4. rocm.py 里 _get_backend_priorities 全文 + 调用点'
grep -n -B6 -A45 'def _get_backend_priorities' "$SITE"/vllm/platforms/rocm.py
echo
echo '### 5. rocm.py 的 get_attn_backend_cls 全文'
grep -n -A60 'def get_attn_backend_cls' "$SITE"/vllm/platforms/rocm.py
echo
echo '### 6. 环境变量的门：VLLM_ROCM_USE_AITER 相关定义'
grep -rn --include='*.py' 'ROCM_USE_AITER' "$SITE"/vllm/envs.py 2>/dev/null | head -20
} > "$OUT" 2>&1
cat "$OUT"
