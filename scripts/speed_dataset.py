"""Log the learned-speed features of a klt_bounded variant, with ground truth, one .npz per episode.

Runs the variant on the host (no sandbox) with `speed_features` switched on, which changes nothing about its output, and
saves for every frame the feature vector the estimator computed (`Solution._speed_features`, NaN before the IMU
calibration ends), the estimator's states, the ground-truth states, the frame times and every tracked corner
(frame, track id, x, y in the processed image, with its camera matrix K). scripts/speed_fit.py trains
and scores speed models on these files.

    .venv-vio/bin/python scripts/speed_dataset.py e5_p05_base3 work/speed/public public
    .venv-vio/bin/python scripts/speed_dataset.py e5_p05_base3 work/speed/sim work/data/sim/*.zip --jobs 6

Sources: `public` (the 10 public episodes), episode hashes, archives or directories, as `viobench debug` takes them.
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solutions" / "klt_bounded"
os.environ.setdefault("CONTA_CACHE", str(ROOT / "work" / "conta-cache"))


def one_episode(job):
    directory, source, out = job
    from viobench.tools.datasets.sources import episodes
    from viobench.tools.runner.direct import PythonSolution

    solution = PythonSolution(directory)
    module = __import__("solution")
    episode = next(iter(episodes([source])))
    config, frames = episode.adapt(solution.resolution)
    estimator = solution.create(config)
    width = len(module.SPEED_FEATURE_NAMES)
    states, features, tracks = [], [], []
    for k, (t, frame, imu) in enumerate(episode.steps(frames)):
        estimator.speed_features = None
        states.append(np.asarray(estimator.step(float(t), frame, imu), dtype=float))
        f = estimator.speed_features
        features.append(np.full(width, np.nan) if f is None else np.asarray(f, dtype=float))
        ids, points = estimator.ids, estimator.previous_points  # every corner observed in this frame, after top-up
        if ids is not None and points is not None and len(ids) == len(points):
            p = points.reshape(-1, 2)
            tracks.append(np.column_stack([np.full(len(p), k), ids, p[:, 0], p[:, 1]]))
    path = Path(out) / f"{episode.hash[:12]}.npz"
    np.savez(path, features=np.array(features), states=np.array(states), truth=episode.ground_truth_state(),
             times=np.asarray(episode.frame_times, dtype=float), names=np.array(module.SPEED_FEATURE_NAMES),
             episode=episode.hash, tracks=np.vstack(tracks) if tracks else np.zeros((0, 4)), K=estimator.K)
    return episode.hash[:12], len(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant", help="a name from solutions/klt_bounded/variants.json")
    parser.add_argument("out", type=Path, help="output directory (one .npz per episode)")
    parser.add_argument("sources", nargs="+", help="`public`, episode hashes, archives or directories")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()

    params = dict(json.loads((SOLUTION / "variants.json").read_text())[args.variant], speed_features=True)
    work = Path(tempfile.mkdtemp(prefix=f"speed-{args.variant}-"))
    shutil.copy(SOLUTION / "solution.py", work / "solution.py")
    (work / "params.json").write_text(json.dumps(params))
    sources = []
    for source in args.sources:
        if source == "public":
            from viobench.episodes import public
            sources += [entry["hash"] for entry in public]
        elif Path(source).is_dir():
            sources += sorted(str(p) for p in Path(source).glob("*.zip"))
        else:
            sources.append(source)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "variant.json").write_text(json.dumps({"variant": args.variant, "params": params}, indent=1))
    with mp.get_context("spawn").Pool(args.jobs) as pool:
        for name, frames in pool.imap(one_episode, [(str(work), s, str(args.out)) for s in sources]):
            print(f"{name}: {frames} frames", flush=True)
    shutil.rmtree(work)


if __name__ == "__main__":
    main()
