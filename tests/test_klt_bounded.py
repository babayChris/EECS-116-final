"""Checks for solutions/klt_bounded: vectorized IMU integration and the known-rotation direction solver.

    .venv-vio/bin/python -m pytest -q tests/test_klt_bounded.py     (or run this file directly)
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(params):
    import json
    bundle = types.ModuleType("solution_bundle")
    bundle.read_text = lambda name: json.dumps(params)
    sys.modules["solution_bundle"] = bundle
    spec = importlib.util.spec_from_file_location("klt_bounded_solution", ROOT / "solutions" / "klt_bounded" / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONFIG = {"fx": 160.0, "fy": 160.0, "cx": 159.5, "cy": 119.5, "width": 320, "height": 240, "T_cam_imu": np.eye(4).tolist()}


def test_vectorized_imu_matches_loop():
    m = load({})
    rng = np.random.default_rng(1)
    a, b = m.Solution(CONFIG), m.Solution(CONFIG)
    for s in (a, b):
        s.R_wb = m.cv2.Rodrigues(np.array([0.3, -0.2, 0.1]))[0]
        s.v = np.array([0.4, -0.1, 0.2]); s.p = np.array([1.0, 2.0, 3.0]); s.last_t = 0.0
        s.gyro_bias = np.array([0.01, -0.02, 0.005])
    t = 0.0
    for _ in range(30):
        rows = []
        for _ in range(10):
            t += 0.005
            rows.append([t, *rng.normal(0, 3, 3) + [0, 0, 9.81], *rng.normal(0, 1.5, 3)])
        rows[3][0] = rows[2][0]  # a repeated timestamp (dt = 0) is skipped by both
        imu = np.array(rows)
        da, db = a._integrate(imu), b._integrate_vectorized(imu)
        assert np.allclose(da, db, atol=1e-9)
    assert np.allclose(a.R_wb, b.R_wb, atol=1e-9) and np.allclose(a.v, b.v, atol=1e-9) and np.allclose(a.p, b.p, atol=1e-9)
    assert np.allclose(a.omega, b.omega) and a.last_t == b.last_t


def direction_errors(noise, outliers, baseline, trials=50):
    m = load({"direction": "known_rotation", "kr_hypotheses": 32, "min_inliers": 10})
    s = m.Solution(CONFIG)
    rng = np.random.default_rng(2)
    errors = []
    for _ in range(trials):
        R = m.cv2.Rodrigues(rng.normal(0, 0.05, 3))[0]          # camera 0 -> camera 1 axes
        c1 = rng.normal(0, 1, 3); c1 *= baseline / np.linalg.norm(c1)  # camera 1 centre in camera 0
        X = np.column_stack([rng.uniform(-3, 3, 40), rng.uniform(-2, 2, 40), rng.uniform(2, 8, 40)])
        Y = X @ R.T - R @ c1
        p0 = X[:, :2] / X[:, 2:] * 160 + [159.5, 119.5]
        p1 = Y[:, :2] / Y[:, 2:] * 160 + [159.5, 119.5] + rng.normal(0, noise, (40, 2))
        p1[:outliers] += rng.uniform(-15, 15, (outliers, 2))
        d = s._direction_known_rotation(p0, p1, R)
        if d is not None:
            errors.append(np.degrees(np.arccos(np.clip(d @ c1 / np.linalg.norm(c1), -1, 1))))
    return np.array(errors)


def test_known_rotation_direction():
    exact = direction_errors(0.0, 0, 0.1)
    assert len(exact) > 40 and exact.max() < 0.01
    robust = direction_errors(0.3, 10, 0.2)                       # 0.3 px noise, 25 % outliers
    assert len(robust) == 50 and np.median(robust) < 2 and np.percentile(robust, 90) < 5


def test_pure_rotation_gives_no_direction():
    m = load({"direction": "known_rotation"})
    s = m.Solution(CONFIG)
    rng = np.random.default_rng(3)
    R = m.cv2.Rodrigues(np.array([0.02, 0.03, -0.01]))[0]
    X = np.column_stack([rng.uniform(-3, 3, 40), rng.uniform(-2, 2, 40), rng.uniform(2, 8, 40)])
    Y = X @ R.T
    p0 = X[:, :2] / X[:, 2:] * 160 + [159.5, 119.5]
    p1 = Y[:, :2] / Y[:, 2:] * 160 + [159.5, 119.5]
    assert s._direction_known_rotation(p0, p1, R) is None


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print(name, "ok")
