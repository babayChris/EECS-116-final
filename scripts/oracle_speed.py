"""Oracle experiment: is the remaining error from step LENGTH (the constant-speed model) or step DIRECTION?

Runs one klt_bounded variant in-process (host, no sandbox, no fuel) on the public episodes, then rebuilds each
estimated trajectory three ways and scores them with the benchmark's own Sim(3) ATE:

  actual            the estimator's output, unchanged
  oracle_speed      estimated direction of every step, true step length   -> what a perfect speed model would give
  oracle_direction  estimated step length, true direction of every step   -> what a perfect direction solver would give

If oracle_speed is much lower than actual, the constant-speed assumption is the bottleneck.
Also reports how much the true speed varies within each episode (coefficient of variation).

    .venv-vio/bin/python scripts/oracle_speed.py e3_persist05            # 10 public episodes, 4 processes
    .venv-vio/bin/python scripts/oracle_speed.py c_best --jobs 6 --episodes 3
"""
import argparse
import json
import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solutions" / "klt_bounded"


def sim3_ate(estimate_xyz, truth_xyz):
    from viobench.tools.runner.metrics import umeyama_sim3

    scale, R, t = umeyama_sim3(estimate_xyz, truth_xyz)
    aligned = (scale * (R @ estimate_xyz.T)).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - truth_xyz) ** 2, axis=1))))


def rebuild(start, lengths, directions):
    """Positions from a start point and per-step lengths along unit directions (zero-length steps stay put)."""
    steps = lengths[:, None] * directions
    return np.vstack([start, start + np.cumsum(steps, axis=0)])


def unit(steps, eps=1e-9):
    norms = np.linalg.norm(steps, axis=1)
    directions = np.zeros_like(steps)
    moving = norms > eps
    directions[moving] = steps[moving] / norms[moving, None]
    return norms, directions, moving


def one_episode(job):
    directory, reference = job
    from viobench.tools.runner.direct import PythonSolution, trace
    from viobench.tools.datasets.sources import episodes

    solution = PythonSolution(directory)
    episode = next(iter(episodes([reference])))
    estimate = np.array([step.pose[:3] for step in trace(solution, episode)])
    truth = episode.ground_truth_state()[:, :3]

    est_len, est_dir, est_moving = unit(np.diff(estimate, axis=0))
    gt_len, gt_dir, _ = unit(np.diff(truth, axis=0))
    oracle_speed = rebuild(estimate[0], np.where(est_moving, gt_len, 0.0), est_dir)  # no direction -> no step
    oracle_direction = rebuild(estimate[0], est_len, gt_dir)

    dt = np.diff(episode.frame_times)
    speed = gt_len / np.maximum(dt, 1e-6)
    return {
        "episode": episode.hash[:8],
        "actual": sim3_ate(estimate, truth),
        "oracle_speed": sim3_ate(oracle_speed, truth),
        "oracle_direction": sim3_ate(oracle_direction, truth),
        "speed_mean_mps": float(speed.mean()),
        "speed_cv": float(speed.std() / max(speed.mean(), 1e-9)),
        "frames_without_direction": int((~est_moving).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", help="a name from solutions/klt_bounded/variants.json")
    parser.add_argument("--jobs", type=int, default=4, help="parallel processes (default 4)")
    parser.add_argument("--episodes", type=int, default=10, help="first N public episodes (default 10)")
    args = parser.parse_args()

    params = json.loads((SOLUTION / "variants.json").read_text())[args.variant]
    work = Path(tempfile.mkdtemp(prefix=f"oracle-{args.variant}-"))
    shutil.copy(SOLUTION / "solution.py", work / "solution.py")
    (work / "params.json").write_text(json.dumps(params))

    from viobench.episodes import public
    jobs = [(str(work), entry["hash"]) for entry in public[: args.episodes]]
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        rows = pool.map(one_episode, jobs)

    header = f"{'episode':9s} {'actual':>7s} {'oracle_speed':>13s} {'oracle_dir':>11s} {'speed m/s':>10s} {'speed CV':>9s}"
    print(f"\nvariant {args.variant}\n{header}\n{'-' * len(header)}")
    for r in rows:
        print(f"{r['episode']:9s} {r['actual']:7.3f} {r['oracle_speed']:13.3f} {r['oracle_direction']:11.3f} "
              f"{r['speed_mean_mps']:10.2f} {r['speed_cv']:9.2f}")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("actual", "oracle_speed", "oracle_direction", "speed_cv")}
    print(f"{'mean':9s} {mean['actual']:7.3f} {mean['oracle_speed']:13.3f} {mean['oracle_direction']:11.3f} "
          f"{'':10s} {mean['speed_cv']:9.2f}")
    gain = 1 - mean["oracle_speed"] / mean["actual"]
    print(f"\nperfect speed would cut mean Sim(3) ATE by {gain:.0%}; "
          f"perfect direction by {1 - mean['oracle_direction'] / mean['actual']:.0%}")
    out = ROOT / "results" / f"oracle-{args.variant}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"variant": args.variant, "params": params, "episodes": rows, "mean": mean}, indent=1))
    print(f"saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
