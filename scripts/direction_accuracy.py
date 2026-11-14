"""Per-frame accuracy of the visual translation direction of a klt_bounded configuration against ground truth.

    .venv-vio/bin/python scripts/direction_accuracy.py <variant name> <episode hash> [...]

Replays the variant on the host, captures each direction `_track` returns (camera k position in camera k-1 axes) and
compares it with the ground-truth relative translation. Also reports how often vision gave a direction, and the
gyro relative-rotation error, since the known-rotation solver depends on it.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONTA_CACHE", str(ROOT / "work" / "conta-cache"))

import numpy as np

from viobench.tools.datasets.sources import episodes
from viobench.tools.geometry import invert, poses_to_matrices
from viobench.tools.runner.direct import PythonSolution, trace

name = sys.argv[1]
spec = json.loads((ROOT / "solutions" / "klt_bounded" / "variants.json").read_text())[name]
directory = ROOT / "work" / "hosttest" / name
directory.mkdir(parents=True, exist_ok=True)
(directory / "solution.py").write_text((ROOT / "solutions" / "klt_bounded" / "solution.py").read_text())
(directory / "params.json").write_text(json.dumps(spec))

for digest in sys.argv[2:]:
    episode = next(episodes(digest))
    solution = PythonSolution(str(directory))
    records = {}
    original = solution.factory._track

    def wrapped(self, frame, R=None, _o=original):
        d = _o(self, frame, R)
        records.setdefault(id(self), {})[self.frame_count - 1] = (None if d is None else np.array(d), None if R is None else np.array(R))
        return d
    solution.factory._track = wrapped
    for _ in trace(solution, episode):
        pass
    solution.factory._track = original
    captured = max(records.values(), key=len)  # the real instance, not the warm-up's throwaway copy
    T = poses_to_matrices(episode.ground_truth_relative())
    errors, rot_errors, given = [], [], 0
    for k, value in captured.items():
        if k == 0 or k >= len(T):
            continue
        rel = invert(T[k - 1]) @ T[k]  # camera k in camera k-1
        d, R = value
        if R is not None:
            dR = R @ rel[:3, :3]  # R maps k-1 -> k axes; rel rotation is k -> k-1
            rot_errors.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
        if d is None:
            continue
        given += 1
        g = rel[:3, 3]
        if np.linalg.norm(g) < 1e-6:
            continue
        errors.append(np.degrees(np.arccos(np.clip(d @ g / np.linalg.norm(d) / np.linalg.norm(g), -1, 1))))
    e = np.array(errors)
    print(f"{name} {digest[:8]}: direction given on {given} frames; angle error median {np.median(e):.1f} deg, "
          f"p75 {np.percentile(e, 75):.1f}, >90 deg on {np.mean(e > 90) * 100:.0f}%; "
          f"gyro relative rotation error median {np.median(rot_errors):.3f} deg")
