#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

mode=${1:-host}
baseline=${2:-klt_imu}
episodes=${3:-1}
case "$mode" in
  host|wasm) ;;
  *) echo "Usage: bash run_local_vio.sh [host|wasm] [klt_imu|dead_reckoning] [episode-count]" >&2; exit 2 ;;
esac
case "$baseline" in
  klt_imu|dead_reckoning) ;;
  *) echo "Choose klt_imu or dead_reckoning." >&2; exit 2 ;;
esac
if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "Episode count must be a positive integer." >&2
  exit 2
fi
if [ ! -f .venv-vio/bin/activate ]; then
  echo "Run bash setup.sh first." >&2
  exit 1
fi
source .venv-vio/bin/activate
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}"
export MPLCONFIGDIR="$PWD/work/matplotlib"
solution="viobench/viobench/solution/examples/python/$baseline"
mkdir -p outputs/local-benchmark
case "$mode" in
  host)
    viobench debug "$solution" --max-episodes "$episodes" \
      -o "outputs/local-benchmark/$baseline-host"
    ;;
  wasm)
    # The first bundle downloads and builds the Python/OpenCV WASM runtime.
    wasm="outputs/local-benchmark/$baseline.wasm"
    viobench bundle "$solution" -o "$wasm"
    viobench check "$wasm"
    viobench evaluate "$wasm" --max-episodes "$episodes" \
      -o "outputs/local-benchmark/$baseline-wasm"
    ;;
esac

