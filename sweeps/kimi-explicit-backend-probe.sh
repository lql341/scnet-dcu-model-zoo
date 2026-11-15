#!/bin/bash
# 显式 --attention-backend 在 0.20 里到底走哪条路径？能否绕过 @if_aiter_supported？
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/explicit.md
{
echo '### 0. selector 模块在哪'
ls -l "$SITE"/vllm/v1/attention/ 2>/dev/null | head -20
find "$SITE"/vllm -name 'selector.py' -o -name 'backend*.py' 2>/dev/null | head -10
echo
echo '### 1. attention_backend 参数在 selector 里怎么用'
SEL=$(find "$SITE"/vllm -path '*attention/selector.py' | head -1)
echo "SELECTOR=$SEL"
grep -n -B4 -A25 'attention_backend' "$SEL" 2>/dev/null | head -70
echo
echo '### 2. get_attn_backend / AttentionBackendEnum 解析'
grep -n -A20 'def get_attn_backend\|def get_backend_class\|def _get_backend' "$SEL" 2>/dev/null | head -60
echo
echo '### 3. @if_aiter_supported 装饰器全文'
grep -n -A22 'def if_aiter_supported' "$SITE"/vllm/_aiter_ops.py
echo
echo '### 4. 显式 backend 时是否还走 validate_configuration'
grep -rn -B3 -A12 'validate_configuration' "$SEL" 2>/dev/null | head -40
} > "$OUT" 2>&1
cat "$OUT"
