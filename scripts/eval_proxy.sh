#!/usr/bin/env bash
# Evaluate built variants on locally simulated episodes and print server-style columns.
#   bash scripts/eval_proxy.sh <episode dir> <parallel> <name=path.wasm>...
# Reports go to outputs/proxy/<name>/; prints mean unaligned ATE, Sim(3) ATE, velocity RMSE, 1-s rotation RPE, peak fuel.
set -euo pipefail
cd "$(dirname "$0")/.."
data=$1; parallel=$2; shift 2
source .venv-vio/bin/activate
mkdir -p outputs/proxy
printf '%s\n' "$@" | xargs -P "$parallel" -I{} sh -c \
  'n=${1%%=*}; w=${1#*=}; viobench evaluate "$w" '"$data"' -o outputs/proxy/$n > outputs/proxy/$n.log 2>&1 || echo "$n: FAILED"' _ {}
python3 - "$@" <<'E'
import json, statistics as st, sys
print(f"{'variant':12s} {'ATE':>6s} {'Sim3':>6s} {'vel':>6s} {'rotRPE':>7s} {'peak':>7s}  per-episode ATE")
for arg in sys.argv[1:]:
    n = arg.split('=', 1)[0]
    try:
        r = json.load(open(f'outputs/proxy/{n}/report.json'))
    except Exception as error:
        print(n, 'no report', error); continue
    m = [e['metrics'] for e in r['episodes']]
    f = lambda k: st.mean(x[k] for x in m)
    print(f"{n:12s} {f('ate_rmse'):6.2f} {f('sim3_ate_rmse'):6.2f} {f('velocity_rmse'):6.2f} {f('rpe_rotation_rmse'):7.4f} "
          f"{r['summary']['fuel_per_frame_max']/1e6:6.2f}M  " + ' '.join(f"{x['ate_rmse']:.2f}" for x in m))
E
