#!/bin/bash -l
# 校验 OmniWeaving 全部产物规格（帧数/fps/时长），并输出 stats.json
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --gres=dcu:1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -J omni-val
#SBATCH -o /public/home/scnethr2601a10/work/omni-val-%j.out
set -uo pipefail
IMG=image.ac.com:5000/dcu/linshp/base/mimo-sglang-bw10000:1.1.1
OUT=/public/home/scnethr2601a10/work/omnirun/repo/official-eval/outputs
export XDG_RUNTIME_DIR=/tmp/xdg-$$; mkdir -p "$XDG_RUNTIME_DIR" /tmp/emptycerts; chmod 700 "$XDG_RUNTIME_DIR"
JID="${SLURM_JOB_ID:-$$}"; PD="podman --root /tmp/pr-$JID --runroot /tmp/prr-$JID --storage-driver overlay --storage-opt ignore_chown_errors=true"
FREE=$(df -Pm /tmp | awk 'NR==2{print int($4/1024)}'); [ "$FREE" -lt 40 ] && { echo "NOT_ENOUGH_SPACE exit 42"; exit 42; }
[ -s /public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz ] && gunzip -c /public/home/scnethr2601a10/zoo-images/mimo-sglang-bw10000_1.1.1.tar.gz | $PD load >/dev/null 2>&1
$PD image exists "$IMG" 2>/dev/null || $PD pull -q --tls-verify=false --cert-dir=/tmp/emptycerts "$IMG" >/dev/null 2>&1
mkdir -p "$OUT/.." ; $PD run --rm --entrypoint=/bin/bash -v "$OUT":/out "$IMG" -c 'python3 - <<PY
import glob, json, os
import imageio.v2 as iio
files = sorted(glob.glob("/out/*.mp4"))
bad, durs, sizes = [], [], []
for f in files:
    try:
        r = iio.get_reader(f); meta = r.get_meta_data(); n = r.count_frames(); r.close()
        fps = float(meta.get("fps", 0)); dur = float(meta.get("duration", 0)); w,h = meta.get("size",(0,0))
        sizes.append(os.path.getsize(f)); durs.append(dur)
        if abs(fps-16) > 0.1 or abs(n-81) > 1 or (w,h) != (832,480):
            bad.append({"file": os.path.basename(f), "fps": fps, "frames": n, "size": [w,h]})
    except Exception as e:
        bad.append({"file": os.path.basename(f), "error": str(e)[:80]})
stats = {"files": len(files), "bytes_total_MB": round(sum(sizes)/1e6,1),
         "dur_s": {"min": min(durs) if durs else None, "max": max(durs) if durs else None},
         "spec_violation_count": len(bad), "violations": bad[:8]}
open("/out/stats.json","w").write(json.dumps(stats, ensure_ascii=False, indent=1))
print(json.dumps(stats, ensure_ascii=False))
PY' 2>&1 | tail -6
