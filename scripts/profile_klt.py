"""Host-side breakdown of where a KLT-style solution spends time per frame, per OpenCV call.

    .venv-vio/bin/python scripts/profile_klt.py <solution dir> <episode hash> [--fuel logs/<hash>/fuel.npy]

Host time is not WASM fuel, but the split between calls points at what drives the expensive frames. With --fuel,
it compares the 1% most expensive frames (by sandbox fuel) with the median ones.
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONTA_CACHE", str(ROOT / "work" / "conta-cache"))

import cv2
import numpy as np

from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.direct import PythonSolution, trace

CALLS = ["cvtColor", "goodFeaturesToTrack", "calcOpticalFlowPyrLK", "findEssentialMat", "recoverPose", "Rodrigues"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("episode")
    ap.add_argument("--fuel", type=Path)
    a = ap.parse_args()
    current = defaultdict(float)
    counts = defaultdict(int)
    for name in CALLS:
        original = getattr(cv2, name)

        def wrapped(*args, _f=original, _n=name, **kw):
            t = time.perf_counter()
            try:
                return _f(*args, **kw)
            finally:
                current[_n] += time.perf_counter() - t
                counts[_n] += 1
        setattr(cv2, name, wrapped)
    per_frame, totals, extra = [], [], []
    episode = next(episodes(a.episode))
    t_prev = time.perf_counter()
    for record in trace(PythonSolution(a.solution), episode):
        now = time.perf_counter()
        per_frame.append({k: current[k] for k in CALLS})
        totals.append(now - t_prev)
        extra.append(counts["findEssentialMat"])
        current.clear()
        t_prev = now
    frames = np.array([[f[k] for k in CALLS] for f in per_frame]) * 1e3
    if a.fuel:
        fuel = np.load(a.fuel)
        n = min(len(fuel), len(frames))
        order = np.argsort(fuel[:n])
        top = order[-max(1, n // 100):]
        mid = order[n // 2 - 50:n // 2 + 50]
        print(f"{'call':24}{'median frames ms':>18}{'top-1% frames ms':>18}")
        for i, k in enumerate(CALLS):
            print(f"{k:24}{frames[mid, i].mean():18.3f}{frames[top, i].mean():18.3f}")
        print(f"{'fuel (M)':24}{fuel[mid].mean() / 1e6:18.1f}{fuel[top].mean() / 1e6:18.1f}")
        corr = [np.corrcoef(frames[12:n, i], fuel[12:n])[0, 1] for i in range(len(CALLS))]
        print("correlation of host call time with sandbox fuel:", {k: round(c, 2) for k, c in zip(CALLS, corr)})
    else:
        for i, k in enumerate(CALLS):
            print(f"{k:24} mean {frames[:, i].mean():.3f} ms  max {frames[:, i].max():.3f} ms")


if __name__ == "__main__":
    sys.exit(main())
