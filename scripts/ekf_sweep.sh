#!/usr/bin/env bash
# Host-screen vio_ekf settings on the local sets: bash scripts/ekf_sweep.sh "P_A=1 P_B=2" ...
cd "$(dirname "$0")/.."
sets=${SETS:-"s3t:work/data/s3test/*.zip ag:work/data/agile10b/g1/*.zip pub:$(tr ' ' ',' < work/public.list)"}
for a in "$@"; do
  line="$a |"
  for s in $sets; do
    n=${s%%:*}; g=${s#*:}; g=${g//,/ }
    r=$(.venv-vio/bin/python scripts/host_native.py native/vio_ekf $(for x in $a; do echo -n "-D $x "; done) --episodes $g --jobs 10 --quiet 2>&1 | tail -1)
    line="$line $n $(echo "$r" | sed -E 's/.*mean ATE ([0-9.]+)  median ([0-9.]+).*/\1 (med \2)/')"
  done
  echo "$line"
done
