"""Sandbox evaluation (viobench evaluate: WASM, fuel) of several builds on several local episode sets.

    .venv-vio/bin/python scripts/eval_sets.py --out results/stage3/local.json --jobs 6 \
        --set s3=work/data/s3 --set agc=work/data/agile_crop name=work/ekf/x.wasm ...

Prints mean unaligned ATE, Sim(3) ATE and peak fuel per set and build, plus the zero-output ATE of each set
(the error of never moving), and writes everything (with per-episode numbers and wasm hashes) to --out.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def zero_ate(paths):
    from viobench.tools.datasets.sources import episodes
    out = []
    for p in paths:
        e = next(episodes(str(p)))
        gt = e.ground_truth_relative()
        out.append(float(np.sqrt((gt[:, :3] ** 2).sum(1).mean())))
    return out


def evaluate(name, wasm, set_name, directory):
    out = ROOT / "outputs" / "sets" / set_name / name
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / ".venv-vio/bin/viobench"), "evaluate", str(wasm)] + ([] if directory == "public" else [str(directory)]) + ["-o", str(out)]
    log = out.with_suffix(".log")
    import os
    env = {**os.environ, "CONTA_CACHE": os.environ.get("CONTA_CACHE", str(ROOT / "work/conta-cache"))}
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    if r.returncode != 0:
        return name, set_name, None
    rep = json.loads((out / "report.json").read_text())
    eps = rep["episodes"]
    return name, set_name, {
        "ate_mean": float(np.mean([e["metrics"]["ate_rmse"] for e in eps])),
        "ate_median": float(np.median([e["metrics"]["ate_rmse"] for e in eps])),
        "sim3_mean": float(np.mean([e["metrics"]["sim3_ate_rmse"] for e in eps])),
        "fuel_max": int(rep["summary"]["fuel_per_frame_max"]),
        "fuel_median": float(rep["summary"]["fuel_per_frame_median"]),
        "completed": int(sum(e["completed"] for e in eps)),
        "episodes": len(eps),
        "per_episode_ate": [round(e["metrics"]["ate_rmse"], 4) for e in eps],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", action="append", required=True, help="name=directory")
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--out")
    p.add_argument("builds", nargs="+", help="name=path.wasm")
    a = p.parse_args()
    sets = dict(s.split("=", 1) for s in a.set)
    builds = dict(b.split("=", 1) for b in a.builds)
    def paths(d):
        if d == "public":
            return (ROOT / "work/public.list").read_text().split()
        return sorted(Path(d).glob("*.zip"))
    zero = {s: zero_ate(paths(d)) for s, d in sets.items()}
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda job: evaluate(*job), [(n, w, s, d) for n, w in builds.items() for s, d in sets.items()]))
    table = {n: {} for n in builds}
    for n, s, r in results:
        table[n][s] = r
    print(f"{'build':22s}" + "".join(f"{s:>22s}" for s in sets))
    print(f"{'zero output':22s}" + "".join(f"{np.mean(zero[s]):>14.3f} m       " for s in sets))
    for n in builds:
        cells = []
        for s in sets:
            r = table[n][s]
            cells.append("        failed        " if r is None else f"{r['ate_mean']:7.3f} m {r['fuel_max'] / 1e6:6.2f} M    ")
        print(f"{n:22s}" + "".join(cells))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        record = {
            "sets": {s: {"directory": d, "episodes": len(zero[s]), "zero_output_ate_mean": float(np.mean(zero[s]))} for s, d in sets.items()},
            "builds": {n: {"wasm": w, "sha256": hashlib.sha256(Path(w).read_bytes()).hexdigest(), "results": table[n]} for n, w in builds.items()},
        }
        Path(a.out).write_text(json.dumps(record, indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
