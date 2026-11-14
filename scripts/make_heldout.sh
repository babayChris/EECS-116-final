#!/usr/bin/env bash
# Generate the held-out generalization set: 10 one-minute episodes, never used for tuning.
# CPU renderer (HYPERLUX_BACKEND=GENERIC) because OptiX is unavailable under WSL. Output: work/data/heldout/<group>/*.zip
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-vio/bin/activate
export HYPERLUX_BACKEND="${HYPERLUX_BACKEND:-GENERIC}" CONTA_CACHE="${CONTA_CACHE:-$PWD/work/conta-cache}" VIOBENCH_SPOOL="$PWD/work/spool"
out=work/data/heldout; mkdir -p "$out" "$VIOBENCH_SPOOL"
# name | scene | seed | extra flags
jobs=(
  "procthor7|ProcTHOR-Train-7|2001|"
  "procthor23|ProcTHOR-Train-23|2002|"
  "procthor57|ProcTHOR-Train-57|2003|"
  "procthor141|ProcTHOR-Train-141|2004|"
  "hssd-a|103997478_171030525|2005|"
  "hssd-b|108736611_177263226|2006|"
  "hssd-c|107734080_175999881|2007|"   # replaces 105515175_173104107: 2880 meshes exceeded memory on the CPU renderer
  "procthor1-seed|ProcTHOR-Train-1|2008|"
  "procthor1-hover3|ProcTHOR-Train-1|2009|--hover 3"
  "procthor1-fov90|ProcTHOR-Train-1|2010|--fov 90 --fov-range 0"
)
for j in "${jobs[@]}"; do
  IFS='|' read -r name scene seed extra <<<"$j"
  ( viobench simulate -o "$out/$name" --scene-names "$scene" --episodes 1 --seed "$seed" --duration 60 --no-video $extra \
      > "$out/$name.log" 2>&1 && echo "$name: ok" || echo "$name: FAILED ($out/$name.log)" ) &
done
wait
