#!/usr/bin/env bash
# Build a variant of native/vio_ekf: bash scripts/build_ekf_variant.sh <name> [P_NAME=value ...]
# Copies the sources to work/ekf/<name>/, prepends the overrides to params.h, bundles work/ekf/<name>.wasm and checks it.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
source .venv-vio/bin/activate
dir=work/ekf/$name; rm -rf "$dir"; mkdir -p "$dir"
cp native/vio_ekf/solution.cpp native/vio_ekf/CMakeLists.txt "$dir/"
{ for kv in "$@"; do echo "#define ${kv%%=*} ${kv#*=}"; done; cat native/vio_ekf/params.h; } > "$dir/params.h"
sed -i "s|\${CMAKE_CURRENT_SOURCE_DIR}/../../viobench|$PWD/viobench|" "$dir/CMakeLists.txt"
viobench bundle "$dir" -o "work/ekf/$name.wasm" > "work/ekf/$name.bundle.log" 2>&1 || { echo "$name: build failed (work/ekf/$name.bundle.log)"; exit 1; }
viobench check "work/ekf/$name.wasm" > "work/ekf/$name.check.log" 2>&1 || { echo "$name: check failed"; exit 1; }
echo "$name: built, $(tail -1 work/ekf/$name.check.log)"
