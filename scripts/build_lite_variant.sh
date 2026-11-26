#!/usr/bin/env bash
# Build a variant of native/vio_lite: bash scripts/build_lite_variant.sh <name> [P_NAME=value ...] -> work/lite/<name>.wasm
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
src=${SRC:-native/vio_lite}
source .venv-vio/bin/activate
dir=work/lite/$name; rm -rf "$dir"; mkdir -p "$dir"
cp $src/solution.cpp $src/CMakeLists.txt "$dir/"
{ for kv in "$@"; do echo "#define ${kv%%=*} ${kv#*=}"; done; cat $src/params.h; } > "$dir/params.h"
sed -i "s|\${CMAKE_CURRENT_SOURCE_DIR}/../../viobench|$PWD/viobench|" "$dir/CMakeLists.txt"
viobench bundle "$dir" -o "work/lite/$name.wasm" > "work/lite/$name.bundle.log" 2>&1 || { echo "$name: build failed (work/lite/$name.bundle.log)"; exit 1; }
viobench check "work/lite/$name.wasm" > "work/lite/$name.check.log" 2>&1 || { echo "$name: check failed"; exit 1; }
echo "$name: built, $(tail -1 work/lite/$name.check.log)"
