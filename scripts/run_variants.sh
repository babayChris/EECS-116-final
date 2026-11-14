#!/usr/bin/env bash
# Bundle and evaluate klt_bounded variants in the WASM sandbox, several in parallel.
#   bash scripts/run_variants.sh [episodes=10] [parallel=5] [variant names...]   (default: all in variants.json)
# Variant directories go to work/variants/<name>/, reports to outputs/variants/<name>-wasm/ (overwritten on rerun).
set -euo pipefail
cd "$(dirname "$0")/.."
episodes=${1:-10}; parallel=${2:-5}; shift $(( $# > 2 ? 2 : $# )) || true
source .venv-vio/bin/activate
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}"
runtime="${VIOBENCH_RUNTIME:-$PWD/viobench/viobench/solution/build/runtime/dist/runtime.wasm}"
[ -f "$runtime" ] || { echo "Build the runtime first: bash run_local_vio.sh wasm klt_imu 1" >&2; exit 1; }
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  mapfile -t names < <(python -c "import json;print('\n'.join(k for k in json.load(open('solutions/klt_bounded/variants.json')) if not k.startswith('_')))")
fi
mkdir -p outputs/variants
for name in "${names[@]}"; do
  dir="work/variants/$name"; rm -rf "$dir"; mkdir -p "$dir"
  cp solutions/klt_bounded/solution.py "$dir/"
  python -c "import json,sys;json.dump(json.load(open('solutions/klt_bounded/variants.json'))[sys.argv[1]],open(sys.argv[2],'w'))" "$name" "$dir/params.json"
  viobench bundle "$dir" -o "work/variants/$name.wasm" --runtime "$runtime" > "work/variants/$name.bundle.log" 2>&1
  viobench check "work/variants/$name.wasm" > "work/variants/$name.check.log" 2>&1 || { echo "$name: check failed (work/variants/$name.check.log)"; }
done
printf '%s\n' "${names[@]}" | xargs -P "$parallel" -I{} sh -c \
  'viobench evaluate work/variants/{}.wasm --max-episodes '"$episodes"' -o outputs/variants/{}-wasm > outputs/variants/{}-wasm.log 2>&1 && echo "{}: done" || echo "{}: FAILED (outputs/variants/{}-wasm.log)"'
