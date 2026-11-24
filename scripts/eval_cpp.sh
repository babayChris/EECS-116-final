#!/usr/bin/env bash
# Sandbox-evaluate built C++ variants: bash scripts/eval_cpp.sh [episodes=10] [parallel=4] names...  -> outputs/variants/<name>-wasm/
set -euo pipefail
cd "$(dirname "$0")/.."
episodes=$1; parallel=$2; shift 2
source .venv-vio/bin/activate
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}"
printf '%s\n' "$@" | xargs -P "$parallel" -I{} sh -c \
  'viobench evaluate work/cpp/{}.wasm --max-episodes '"$episodes"' -o outputs/variants/{}-wasm > outputs/variants/{}-wasm.log 2>&1 && echo "{}: done" || echo "{}: FAILED"'
