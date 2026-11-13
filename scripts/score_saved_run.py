"""Re-score the committed trajectory using the pinned upstream scoring code.

Downloads the one public episode if it is absent from CONTA_CACHE. Does not
re-run the estimator, generate ground truth, or measure WASM fuel.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONTA_CACHE", str(ROOT / "work" / "conta-cache"))

import numpy as np

from viobench.tools.datasets.sources import resolve
from viobench.tools.datasets.storage import read_episode
from viobench.tools.runner.metrics import evaluate_trajectory
from viobench.tools.runner.score import load_estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Use an already downloaded public episode archive.")
    parser.add_argument("--output", type=Path, help="Save the recomputed metrics as JSON (default: print only).")
    args = parser.parse_args()
    run_dir = ROOT / "results" / "klt-imu-host-first-public"
    recorded = json.loads((run_dir / "host-results.json").read_text())
    archive = args.archive or resolve([recorded["episode"]])[0]
    episode = read_episode(archive, frames="none")
    if episode.hash != recorded["episode"]["hash"]:
        raise SystemExit("The archive does not match the recorded public episode hash.")
    estimates = load_estimates(run_dir / "estimate.tum", episode.frame_times)
    truth = episode.ground_truth_state()
    metrics = evaluate_trajectory(truth, estimates, episode.frame_times).as_dict()
    mismatches = [name for name, value in recorded["metrics"].items()
                  if not np.isclose(metrics[name], value, rtol=1e-7, atol=1e-9, equal_nan=True)]
    if mismatches:
        raise SystemExit("Re-scoring differs from the recorded metrics: " + ", ".join(mismatches))
    stationary = np.zeros_like(truth)
    stationary[:, 6] = 1.0
    stationary_metrics = evaluate_trajectory(truth, stationary, episode.frame_times)
    solution = ROOT / "viobench" / "viobench" / "solution" / "examples" / "python" / "klt_imu" / "solution.py"
    digest = hashlib.sha256(solution.read_bytes()).hexdigest()
    result = {
        "matches_recorded_metrics": True,
        "upstream_solution_matches_recorded_sha256": digest == recorded["solution_sha256"],
        "episode_hash": episode.hash,
        "metrics": metrics,
        "constant_pose_sim3_ate_m": stationary_metrics.sim3_ate_rmse,
        "fuel_per_frame_max": None,
        "note": "Scores saved host predictions; no estimator rerun and no WASM fuel measurement.",
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()

