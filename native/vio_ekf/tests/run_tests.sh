#!/usr/bin/env bash
# Numerical checks of the vio_ekf Jacobians (host compiler): bash native/vio_ekf/tests/run_tests.sh
set -euo pipefail
cd "$(dirname "$0")/.."
inc="-I../../viobench/viobench/solution/runtime/native -I."
out=$(mktemp -d)
g++ -O1 -std=c++17 $inc tests/jacobian_test.cpp -o "$out/jt" && "$out/jt" | tail -1
g++ -O1 -std=c++17 $inc tests/phi_test.cpp -o "$out/pt" && "$out/pt" | grep "max err"
rm -rf "$out"
