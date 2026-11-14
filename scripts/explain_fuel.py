"""Explain per-frame sandbox fuel of a klt_bounded variant from what each frame did (replayed on the host).

    .venv-vio/bin/python scripts/explain_fuel.py <variant name> <episode hash>

Replays the variant's solution on the host, recording per frame: tracks entering LK, tracks surviving, points given
to findEssentialMat, its inliers, whether recoverPose ran, corners detected and IMU rows. Then fits sandbox fuel
(outputs/variants/<name>-wasm/logs/<hash>/fuel.npy) by least squares on those counts and prints the costliest frames.
The host replay follows the same decisions as the sandbox only where host and WASM OpenCV agree; treat as diagnosis.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CONTA_CACHE", str(ROOT / "work" / "conta-cache"))

import cv2
import numpy as np

from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.direct import PythonSolution, trace

name, digest = sys.argv[1], sys.argv[2]
rec = {}
orig = {n: getattr(cv2, n) for n in ["calcOpticalFlowPyrLK", "findEssentialMat", "recoverPose", "goodFeaturesToTrack"]}


def lk(prev, nxt, pts, *a, **k):
    out = orig["calcOpticalFlowPyrLK"](prev, nxt, pts, *a, **k)
    rec["lk_in"] = len(pts); rec["lk_ok"] = int(out[1].sum())
    return out


def fe(p0, p1, *a, **k):
    out = orig["findEssentialMat"](p0, p1, *a, **k)
    rec["e_pts"] = len(p0); rec["e_inl"] = 0 if out[1] is None else int(out[1].sum())
    rec["e_sols"] = 0 if out[0] is None else out[0].shape[0] // 3
    return out


def rp(*a, **k):
    rec["rp"] = 1
    return orig["recoverPose"](*a, **k)


def gf(*a, **k):
    out = orig["goodFeaturesToTrack"](*a, **k)
    rec["corners"] = 0 if out is None else len(out)
    return out


cv2.calcOpticalFlowPyrLK, cv2.findEssentialMat, cv2.recoverPose, cv2.goodFeaturesToTrack = lk, fe, rp, gf
fields = ["lk_in", "lk_ok", "e_pts", "e_inl", "e_sols", "rp", "corners", "imu"]
rows = []
for r in trace(PythonSolution(str(ROOT / "work" / "variants" / name)), next(episodes(digest))):
    rows.append([rec.get(f, 0) for f in fields[:-1]] + [len(r.imu) if hasattr(r, "imu") and r.imu is not None else 0])
    rec.clear()
A = np.array(rows, float)
fuel = np.load(ROOT / "outputs" / "variants" / f"{name}-wasm" / "logs" / digest / "fuel.npy")[:len(A)] / 1e6
s = slice(12, None)
X = np.column_stack([np.ones(len(A)), A])[s]
coef, *_ = np.linalg.lstsq(X, fuel[s], rcond=None)
pred = X @ coef
print("fit R^2 = %.3f" % (1 - ((fuel[s] - pred) ** 2).sum() / ((fuel[s] - fuel[s].mean()) ** 2).sum()))
print("M fuel per unit:", {k: round(v, 4) for k, v in zip(["const"] + fields, coef)})
top = np.argsort(fuel)[-8:][::-1]
print("frame  fuel(M)  " + "  ".join(fields))
for i in top:
    print(f"{i:5d}  {fuel[i]:7.1f}  " + "  ".join(f"{int(v):>{len(f)}}" for v, f in zip(A[i], fields)))
med = np.argsort(fuel)[len(fuel) // 2]
print(f"median frame {med}: {fuel[med]:.1f}M ", dict(zip(fields, A[med].astype(int))))
print("E frames: %d of %d; mean fuel with E %.1fM, without %.1fM" % (A[s, 2].astype(bool).sum(), len(A) - 12,
      fuel[s][A[s, 2] > 0].mean(), fuel[s][A[s, 2] == 0].mean()))
