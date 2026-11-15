#!/bin/bash
# 容器内：能否让 0.20 的 aiter 初始化成功（缺 rocminfo 是已知阻塞）
set -uo pipefail
SITE=$(python3 -c 'import sysconfig;print(sysconfig.get_paths()["purelib"])')
OUT=/gate/rocminfo.md
{
echo '### 0. rocminfo 在哪'
for p in /opt/dtk/bin/rocminfo /opt/dtk/.hyhal/rocm_smi/bin/rocminfo /usr/bin/rocminfo; do
  [ -x "$p" ] && echo "FOUND $p" || echo "missing $p"
done
which rocminfo 2>/dev/null || echo "PATH 里没有 rocminfo"
find /opt/dtk -maxdepth 4 -name 'rocminfo*' -type f 2>/dev/null | head -5
echo
echo '### 1. aiter 怎么探测 arch（看它到底调用什么）'
grep -rnE 'rocminfo|get_gpu_arch|arch_node|gfx' "$SITE"/aiter/jit/utils/*.py 2>/dev/null | grep -iE 'rocminfo|def get_gpu_arch|subprocess|check_output' | head -15
echo '--- 找 rocminfo 调用点 ---'
grep -rn 'rocminfo' "$SITE"/aiter --include=*.py 2>/dev/null | head -10
echo
echo '### 2. 直接 import aiter（现状）'
python3 -c 'import aiter; print("aiter OK", aiter.__file__)' 2>&1 | tail -4
echo
echo '### 3. 用假 rocminfo / 环境变量绕过'
mkdir -p /tmp/fakebin
cat > /tmp/fakebin/rocminfo <<'SH'
#!/bin/bash
echo "  Name:                    gfx936"
SH
chmod +x /tmp/fakebin/rocminfo
PATH=/tmp/fakebin:$PATH python3 -c 'import aiter; print("aiter OK with fake rocminfo")' 2>&1 | tail -5
echo
echo '### 4. 直接问 torch 能不能拿到 arch'
python3 -c 'import torch; print("arch list:", torch.cuda.get_arch_list() if torch.cuda.is_available() else "cuda unavailable"); print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")' 2>&1 | tail -5
echo
echo '### 5. 如果 aiter 起来了，MLA 候选集变成什么'
PATH=/tmp/fakebin:$PATH python3 - <<'PY' 2>&1 | tail -15
try:
    from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
    print('is_aiter_found_and_supported =', is_aiter_found_and_supported())
    print('is_mla_enabled =', rocm_aiter_ops.is_mla_enabled())
    from vllm.platforms.rocm import _get_backend_priorities
    print('priorities(use_mla=True) =', [b.name for b in _get_backend_priorities(True, False)])
except Exception as e:
    import traceback; traceback.print_exc()
PY
} > "$OUT" 2>&1
cat "$OUT"
