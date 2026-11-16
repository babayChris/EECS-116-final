"""Fit and score learned speed models on episodes logged by scripts/speed_dataset.py.

For each held-out episode the model predicts the speed of every frame from that frame's features; the estimator's own
rotation and translation direction are kept and each step is rescaled to predicted speed x frame interval, then the
chained trajectory is scored with the benchmark's Sim(3) ATE. Rows: the estimator as logged, a constant step (what
the estimator does, re-chained), the model, and the true step length (the upper bound for any speed model).

    .venv-vio/bin/python scripts/speed_fit.py --cv work/speed/public          # leave one episode out (diagnostic)
    .venv-vio/bin/python scripts/speed_fit.py --train work/speed/sim --test work/speed/public
"""
import argparse
import json
from pathlib import Path

import numpy as np

from viobench.tools.geometry import invert, matrices_to_poses, poses_to_matrices
from viobench.tools.runner.metrics import evaluate_trajectory

LOG_EPS = 1e-3  # added before taking logs of flow rates (normalised units per second)


def load_one(path):
    d = np.load(path)
    states, truth, times = d["states"], d["truth"], d["times"]
    Te, Tg = poses_to_matrices(states[:, :7]), poses_to_matrices(truth[:, :7])
    rel_e = np.einsum("nij,njk->nik", np.linalg.inv(Te[:-1]), Te[1:])
    rel_g = np.einsum("nij,njk->nik", np.linalg.inv(Tg[:-1]), Tg[1:])
    dt = np.diff(times)
    return {
        "name": Path(path).stem[:8], "names": [str(n) for n in d["names"]], "X": d["features"][1:], "dt": dt,
        "speed": np.linalg.norm(rel_g[:, :3, 3], axis=1) / dt, "rel_e": rel_e, "rel_g": rel_g, "states": states,
        "truth": truth, "times": times,
    }


def load(directories):
    return [load_one(path) for directory in directories for path in sorted(Path(directory).glob("*.npz"))]


def column_spec(names):
    """Model inputs as (op, i, j) over the logged features: every feature as is, the log of every flow-related one
    (has_flow included, as in the measured results) and the x/y specific force magnitude. Solution._learned_speed
    evaluates the same list from speed_model.json, so training and the bundled estimator cannot disagree."""
    spec = [("raw", i, -1) for i in range(len(names))]
    labels = list(names)
    for i, name in enumerate(names):
        if "flow" in name:
            spec.append(("log", i, -1))
            labels.append(f"log_{name}")
    for prefix in ("", "ema0.5_", "ema2.0_"):
        spec.append(("hypot", names.index(f"{prefix}acc_x"), names.index(f"{prefix}acc_y")))
        labels.append(f"{prefix}acc_xy")
    return spec, labels


def expand(X, names):
    spec, labels = column_spec(names)
    columns = [X[:, i] if op == "raw" else np.log(X[:, i] + LOG_EPS) if op == "log" else np.hypot(X[:, i], X[:, j])
               for op, i, j in spec]
    return np.column_stack(columns), labels


class Ridge:
    def __init__(self, alpha=1.0, shrink=1.0):
        self.alpha, self.shrink = alpha, shrink

    def speed(self, X):
        """Deployed prediction: clipped at zero, then shrunk toward the training mean (as Solution._learned_speed)."""
        return self.shrink * np.clip(self.predict(X), 0.0, None) + (1.0 - self.shrink) * self.y_mean

    def export(self, names, path, trained_on):
        spec, labels = column_spec(names)
        path.write_text(json.dumps({
            "feature_names": list(names), "columns": [list(c) for c in spec], "column_labels": labels,
            "log_eps": LOG_EPS, "mean": self.mean.tolist(), "std": self.std.tolist(), "weights": self.w[:-1].tolist(),
            "intercept": float(self.w[-1]), "shrink": self.shrink, "train_mean_speed": float(self.y_mean),
            "alpha": self.alpha, "trained_on": trained_on,
        }, indent=1))

    def fit(self, X, y):
        self.y_mean = float(y.mean())
        self.mean, self.std = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([(X - self.mean) / self.std, np.ones((len(X), 1))])
        penalty = self.alpha * np.eye(Z.shape[1])
        penalty[-1, -1] = 0.0
        self.w = np.linalg.solve(Z.T @ Z + penalty, Z.T @ y)
        return self

    def predict(self, X):
        return np.hstack([(X - self.mean) / self.std, np.ones((len(X), 1))]) @ self.w


def rows(episode):
    valid = np.all(np.isfinite(episode["X"]), axis=1)
    return valid


def chain_score(episode, lengths):
    """Sim(3) ATE of the trajectory with the estimator's rotations and directions and the given step lengths."""
    rel = episode["rel_e"].copy()
    norms = np.linalg.norm(rel[:, :3, 3], axis=1)
    moving = norms > 1e-9
    rel[moving, :3, 3] *= (lengths[moving] / norms[moving])[:, None]
    T = np.empty((len(rel) + 1, 4, 4))
    T[0] = np.eye(4)
    for k in range(len(rel)):
        T[k + 1] = T[k] @ rel[k]
    estimate = np.zeros_like(episode["truth"])
    estimate[:, :7] = matrices_to_poses(T)
    return float(evaluate_trajectory(episode["truth"], estimate, episode["times"]).sim3_ate_rmse)


def design(episodes):
    names = episodes[0]["names"]
    X = np.vstack([e["X"][rows(e)] for e in episodes])
    y = np.concatenate([e["speed"][rows(e)] for e in episodes])
    return expand(X, names)[0], y


def score(model, episode):
    names = episode["names"]
    valid = rows(episode)
    predicted = np.zeros(len(episode["dt"]))
    predicted[valid] = model.speed(expand(episode["X"][valid], names)[0])
    norms = np.linalg.norm(episode["rel_e"][:, :3, 3], axis=1)
    truth_speed = episode["speed"]
    r = np.corrcoef(predicted[valid], truth_speed[valid])[0, 1]
    return {
        "episode": episode["name"],
        "logged": float(evaluate_trajectory(episode["truth"], episode["states"], episode["times"]).sim3_ate_rmse),
        "constant": chain_score(episode, np.where(norms > 1e-9, 1.0, 0.0)),
        "model": chain_score(episode, predicted * episode["dt"]),
        "true_length": chain_score(episode, truth_speed * episode["dt"]),
        "speed_rmse": float(np.sqrt(np.mean((predicted[valid] - truth_speed[valid]) ** 2))),
        "speed_r": float(r),
    }


def report(results, title):
    keys = ("logged", "constant", "model", "true_length", "speed_rmse", "speed_r")
    print(f"\n{title}\n{'episode':9s}" + "".join(f"{k:>12s}" for k in keys))
    for r in results:
        print(f"{r['episode']:9s}" + "".join(f"{r[k]:12.3f}" for k in keys))
    print(f"{'mean':9s}" + "".join(f"{np.mean([r[k] for r in results]):12.3f}" for k in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cv", nargs="+", help="leave-one-episode-out over these logged directories")
    parser.add_argument("--train", nargs="+")
    parser.add_argument("--test", nargs="+")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--shrink", type=float, default=1.0,
                        help="weight of the prediction against the training mean speed (1: prediction only)")
    parser.add_argument("--export", type=Path, help="fit on --train (or --cv) episodes and write speed_model.json here")
    parser.add_argument("--save", type=Path, help="write the results as JSON")
    args = parser.parse_args()

    results = []
    if args.cv:
        episodes = load(args.cv)
        for i, held in enumerate(episodes):
            model = Ridge(args.alpha, args.shrink).fit(*design(episodes[:i] + episodes[i + 1:]))
            results.append(score(model, held))
        report(results, f"leave-one-episode-out, ridge alpha {args.alpha}, shrink {args.shrink}, {len(episodes)} episodes")
        train = episodes
    else:
        train = load(args.train)
        if args.test:
            model = Ridge(args.alpha, args.shrink).fit(*design(train))
            results = [score(model, e) for e in load(args.test)]
            report(results, f"trained on {len(train)} episodes, ridge alpha {args.alpha}, shrink {args.shrink}")
    full = Ridge(args.alpha, args.shrink).fit(*design(train))
    labels = column_spec(train[0]["names"])[1]
    order = np.argsort(-np.abs(full.w[:-1]))
    print("\nlargest standardised weights (full fit): " + ", ".join(f"{labels[j]} {full.w[j]:+.3f}" for j in order[:10]))
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        full.export(train[0]["names"], args.export, sorted(e["name"] for e in train))
        print(f"wrote {args.export} (trained on {len(train)} episodes)")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({"results": results, "alpha": args.alpha, "shrink": args.shrink}, indent=1))


if __name__ == "__main__":
    main()
