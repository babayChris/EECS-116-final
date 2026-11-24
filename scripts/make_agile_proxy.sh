#!/usr/bin/env bash
# Simulate "agile" proxy episodes for the stage 2 setting (a guess at it: the real generator is not published).
#   bash scripts/make_agile_proxy.sh [episodes=4] [duration_s=10] [seed=8801] [out=work/data/agile]
# Applies scripts/agile_autopilot_proxy.patch (faster cruise, harder acceleration, jerk and yaw rate for the
# simulator's autopilot) to the pinned viobench, runs `viobench simulate`, and reverts the patch afterwards, even on
# failure. Episodes are written in batches of 4 (all four appear at the end of a batch).
# Arch Linux needs `minizip` and `vulkan-headers` for the renderer build. Rendering is bound by one CPU thread
# (about 0.3 frames/s per episode here), so a 10 s episode takes about 11 minutes; there is no timeout on purpose.
set -euo pipefail
cd "$(dirname "$0")/.."
episodes=${1:-4}; duration=${2:-10}; seed=${3:-8801}; out=${4:-work/data/agile}
patch="$PWD/scripts/agile_autopilot_proxy.patch"
git -C viobench apply "$patch"
trap 'git -C viobench apply -R "$patch"' EXIT
source .venv-vio/bin/activate
export CPATH="${CPATH:+$CPATH:}/usr/include/minizip"
[ -f /usr/lib/libcblas.so.3 ] && export LD_PRELOAD=/usr/lib/libcblas.so.3  # the compiled environment needs cblas_sgemm
export CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}" VIOBENCH_SPOOL="${VIOBENCH_SPOOL:-$PWD/work/spool-agile}"
mkdir -p "$out" "$VIOBENCH_SPOOL"
viobench simulate -o "$out" --scene-names ProcTHOR-Train-1 --episodes "$episodes" --seed "$seed" --duration "$duration" --no-video
echo "episodes in $out; score builds with: bash scripts/eval_proxy.sh $out 8 name=path.wasm ..."
