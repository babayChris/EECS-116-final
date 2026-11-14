#!/usr/bin/env bash
# Evaluate already-built .wasm files (the same artifacts measured on the public set) on the held-out episodes.
#   bash scripts/eval_heldout.sh [parallel=6] names...
# A name is an upstream baseline (minimal, dead_reckoning, klt_imu: outputs/local-benchmark/<name>.wasm) or a
# klt_bounded variant (work/variants/<name>.wasm). Reports: $HELDOUT_OUT/<name>-wasm/ (default outputs/heldout).
# HELDOUT_GLOB selects episodes (default: all of work/data/heldout).
set -euo pipefail
cd "$(dirname "$0")/.."
parallel=${1:-6}; shift
source .venv-vio/bin/activate
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}"
episodes=(${HELDOUT_GLOB:-work/data/heldout/*/*.zip})
out=${HELDOUT_OUT:-outputs/heldout}
[ ${#episodes[@]} -gt 0 ] || { echo "No held-out episodes: run scripts/make_heldout.sh" >&2; exit 1; }
mkdir -p "$out"
for name in "$@"; do
  wasm="work/variants/$name.wasm"; [ -f "$wasm" ] || wasm="outputs/local-benchmark/$name.wasm"
  [ -f "$wasm" ] || { echo "$name: no wasm" >&2; exit 1; }
  echo "$name $wasm"
done | xargs -P "$parallel" -L1 sh -c 'viobench evaluate "$1" '"${episodes[*]}"' -o '"$out"'/"$0"-wasm > '"$out"'/"$0"-wasm.log 2>&1 && echo "$0: done" || echo "$0: FAILED"'
