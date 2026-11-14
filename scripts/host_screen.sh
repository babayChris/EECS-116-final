#!/usr/bin/env bash
# Quick host-mode accuracy screen of klt_bounded variants on the public episodes (no fuel; host libraries).
#   bash scripts/host_screen.sh [episodes=10] [parallel=8] names...
# Prints mean Sim(3) ATE per variant. Sandbox runs (scripts/run_variants.sh) remain the measure.
# HOST_EPISODES="paths..." evaluates those episodes instead of the first <episodes> public ones; HOST_TAG names the logs.
set -euo pipefail
cd "$(dirname "$0")/.."
episodes=$1; parallel=$2; shift 2
source .venv-vio/bin/activate
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}" MPLCONFIGDIR="$PWD/work/matplotlib"
for name in "$@"; do
  d="work/hosttest/$name"; rm -rf "$d"; mkdir -p "$d"
  cp solutions/klt_bounded/solution.py "$d/"
  python -c "import json,sys;json.dump(json.load(open('solutions/klt_bounded/variants.json'))[sys.argv[1]],open(sys.argv[2],'w'))" "$name" "$d/params.json"
done
tag=${HOST_TAG:-public}
if [ -n "${HOST_EPISODES:-}" ]; then select="$HOST_EPISODES"; else select="--max-episodes $episodes"; fi
printf '%s\n' "$@" | xargs -P "$parallel" -I{} sh -c 'viobench debug work/hosttest/{} '"$select"' > work/hosttest/{}.'"$tag"'.log 2>&1 || echo "{}: FAILED"'
for name in "$@"; do
  python - "$name" "$tag" <<'PY'
import re, sys
name, tag = sys.argv[1], sys.argv[2]
vals = [float(m) for m in re.findall(r"Sim\(3\) ATE ([0-9.]+) m", open(f"work/hosttest/{name}.{tag}.log").read())]
unal = [float(m) for m in re.findall(r"  ATE ([0-9.]+) m", open(f"work/hosttest/{name}.{tag}.log").read())]
unal.sort()
print(f"{name:26} episodes {len(vals):2d}  mean Sim(3) ATE {sum(vals)/max(len(vals),1):.3f} m  median unaligned {unal[len(unal)//2] if unal else float('nan'):8.1f} m  per-episode " + " ".join(f"{v:.2f}" for v in vals))
PY
done
