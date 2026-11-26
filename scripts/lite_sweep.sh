#!/usr/bin/env bash
# Host-screen settings of a native solution on local sets: SRC=native/vio_lite SETS="name:glob,glob ..." bash scripts/lite_sweep.sh "P_A=1 P_B=2" ...
cd "$(dirname "$0")/.."
src=${SRC:-native/vio_lite}
sets=${SETS:-"tune:work/data/s3/s3-ProcTHOR-Train-12-*.zip,work/data/s3/s3-ProcTHOR-Train-45-*.zip tune3:work/data/s3/s3-ProcTHOR-Train-12-*.zip@0:45:3,work/data/s3/s3-ProcTHOR-Train-45-*.zip@0:45:3 agc:work/data/agile_crop/*.zip"}
for a in "$@"; do
  line="$a |"
  for s in $sets; do
    n=${s%%:*}; g=${s#*:}; g=${g//,/ }
    r=$(.venv-vio/bin/python scripts/host_native.py $src $(for x in $a; do echo -n "-D $x "; done) --episodes $g --jobs ${JOBS:-10} --quiet 2>&1 | tail -1)
    line="$line $n $(echo "$r" | sed -E 's/.*mean ATE ([0-9.]+)  median ([0-9.]+)  sim3 ([0-9.]+).*/\1 (s3 \3)/')"
  done
  echo "$line"
done
