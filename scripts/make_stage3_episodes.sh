#!/usr/bin/env bash
# Generate stage-3-like episodes (see scripts/make_stage3_episodes.py). Applies scripts/stage3_autopilot.patch to the
# pinned viobench while the jobs run and reverts it afterwards, even on failure.
#   bash scripts/make_stage3_episodes.sh <out> <duration_s> "<scene>|<seed>" ...   (one parallel job of 4 flights each)
set -euo pipefail
cd "$(dirname "$0")/.."
out=$1; duration=$2; shift 2
patch="$PWD/scripts/stage3_autopilot.patch"
git -C viobench apply "$patch"
trap 'git -C viobench apply -R "$patch"' EXIT
source .venv-vio/bin/activate
export CPATH="${CPATH:+$CPATH:}/usr/include/minizip"
[ -f /usr/lib/libcblas.so.3 ] && export LD_PRELOAD=/usr/lib/libcblas.so.3
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}"
mkdir -p "$out"
for job in "$@"; do
  IFS='|' read -r scene seed <<<"$job"
  ( python scripts/make_stage3_episodes.py -o "$out" --scene "$scene" --episodes 4 --seed "$seed" --duration "$duration" \
      --spool "$PWD/work/spool-s3/$seed" > "$out/$scene-$seed.log" 2>&1 && echo "$scene $seed: ok" || echo "$scene $seed: FAILED" ) &
  sleep 60   # stagger the JIT compile: the first job builds the shared module
done
wait
