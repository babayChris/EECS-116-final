#!/usr/bin/env bash
# Build a variant of the C++ port: bash scripts/build_cpp_variant.sh <name> [P_NAME=value ...]
# Copies native/klt_bounded to work/cpp/<name>/, prepends the overrides to params.h, bundles work/cpp/<name>.wasm and checks it.
set -euo pipefail
cd "$(dirname "$0")/.."
name=$1; shift
source .venv-vio/bin/activate
dir=work/cpp/$name; rm -rf "$dir"; mkdir -p "$dir"
cp native/klt_bounded/solution.cpp native/klt_bounded/speed_model.h native/klt_bounded/CMakeLists.txt "$dir/"
{ for kv in "$@"; do echo "#define ${kv%%=*} ${kv#*=}"; done; cat native/klt_bounded/params.h; } > "$dir/params.h"
sed -i "s|\${CMAKE_CURRENT_SOURCE_DIR}/../../viobench|$PWD/viobench|" "$dir/CMakeLists.txt"
viobench bundle "$dir" -o "work/cpp/$name.wasm" > "work/cpp/$name.bundle.log" 2>&1 || { echo "$name: build failed"; exit 1; }
viobench check "work/cpp/$name.wasm" > "work/cpp/$name.check.log" 2>&1 || { echo "$name: check failed"; exit 1; }
echo "$name: built, check $(tail -1 work/cpp/$name.check.log), floor $(grep 'step 4' work/cpp/$name.check.log | awk '{print $4}')"
