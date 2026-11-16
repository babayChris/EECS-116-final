"""Offline prototype: relative step length from triangulated track depths (monocular scale propagation).

Works on episodes logged by scripts/speed_dataset.py (tracked corners + the estimator's states). The estimator's own
rotation R and translation direction d between frames are taken as given; only the step length s is estimated.

Every track carries an inverse depth rho (1 / depth along the camera axis, in units of 1 / u, where u is an arbitrary
length unit fixed at the start) with a variance. For a corner seen at bearing x0 in frame k-1 and x1 in frame k,

    x1 ~ project(R x0 - (s * rho) R d)

so each track measures lambda = s * rho (linearised least squares on its image displacement). Tracks whose depth is
already well known give s = lambda / rho; their weighted median is the frame's step length. Then every track's rho is
updated with lambda / s (a one-dimensional Kalman filter) and propagated to frame k. New tracks start from a weak prior
at the median inverse depth. Without usable tracks the previous step length is kept. The unit u, and so the speed
scale, drifts slowly; Sim(3) alignment removes one global scale.

    .venv-vio/bin/python scripts/depth_scale.py work/speed/public
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import speed_fit  # noqa: E402


def propagate_scale(episode, tracks, K, sigma_px=0.15, mature=0.3, process=0.02, min_mature=5, motion="rel_e"):
    """Per-frame step length (units u per frame) and the number of tracks used, for the logged episode."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    sigma = sigma_px / fx  # normalised image units
    frames = {}
    for k in np.unique(tracks[:, 0]).astype(int):
        rows = tracks[tracks[:, 0] == k]
        frames[k] = dict(zip(rows[:, 1].astype(int), np.column_stack([(rows[:, 2] - cx) / fx, (rows[:, 3] - cy) / fy])))
    rel = episode[motion]
    n = len(rel)
    steps, used = np.full(n, np.nan), np.zeros(n, int)
    rho, var = {}, {}  # per track id, in the frame of its last observation
    s_previous = 1.0
    reference = 1.0  # median inverse depth of the current tracks: the yardstick for maturity
    for j in range(n):  # motion from frame j to frame j + 1
        a, b_frame = frames.get(j), frames.get(j + 1)
        t = rel[j][:3, 3]
        length = np.linalg.norm(t)
        if a is None or b_frame is None or length < 1e-12:
            steps[j] = 0.0 if length < 1e-12 else s_previous
            continue
        R = rel[j][:3, :3].T  # camera j axes -> camera j+1 axes
        bvec = R @ (t / length)
        common = [i for i in b_frame if i in a]
        lam, lam_var, r0z = {}, {}, {}
        for i in common:
            r0 = R @ np.array([a[i][0], a[i][1], 1.0])
            p = r0[:2] / r0[2]
            Jb = (bvec[:2] - p * bvec[2]) / r0[2]
            g = float(Jb @ Jb)
            if g < 1e-8:
                continue
            lam[i] = float((p - b_frame[i]) @ Jb) / g
            lam_var[i] = sigma * sigma / g
            r0z[i] = r0[2]
        # Maturity and weights are judged against the scene's median inverse depth, not each track's own estimate:
        # a relative test (sqrt(var) < mature * rho[i]) admits tracks whose rho is overestimated first, which biases
        # s = lambda / rho low; new tracks then inherit an even larger rho and the scale shrinks without bound.
        estimates, weights = [], []
        for i in lam:
            if i in rho and rho[i] > 0 and np.sqrt(var[i]) < mature * reference:
                estimates.append(lam[i] / rho[i])
                weights.append(1.0 / (lam_var[i] + (s_previous ** 2) * var[i]))
        if len(estimates) >= min_mature:
            order = np.argsort(estimates)
            cumulative = np.cumsum(np.asarray(weights)[order])
            s = float(np.asarray(estimates)[order][np.searchsorted(cumulative, cumulative[-1] / 2)])
            s = max(s, 0.0)
            used[j] = len(estimates)
        else:
            s = s_previous
        steps[j] = s
        known = [rho[i] for i in rho if np.sqrt(var[i]) < mature * reference]
        prior = float(np.median(known)) if known else reference
        new_rho, new_var = {}, {}
        for i in lam:
            r, v = (rho[i], var[i]) if i in rho else (prior, (2.0 * prior) ** 2)
            if s > 1e-9:
                z, zv = lam[i] / s, lam_var[i] / (s * s)
                innovation = z - r
                if innovation * innovation < 9.0 * (v + zv) or i not in rho:
                    gain = v / (v + zv)
                    r, v = r + gain * innovation, (1.0 - gain) * v
            r = max(r, 1e-3 * prior)
            denominator = r0z[i] - s * bvec[2] * r
            if denominator <= 1e-3:
                continue  # passed behind or onto the camera: drop the track
            new_rho[i] = r / denominator
            new_var[i] = v * (r0z[i] / denominator ** 2) ** 2 + (process * new_rho[i]) ** 2
        rho, var = new_rho, new_var
        if rho:
            reference = float(np.median(list(rho.values())))
        s_previous = s if s > 0 else s_previous
    return steps, used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logged", nargs="+")
    parser.add_argument("--sigma-px", type=float, default=0.15)
    parser.add_argument("--process", type=float, default=0.02)
    parser.add_argument("--oracle-motion", action="store_true", help="diagnostic: true rotation and direction instead of the estimator's")
    args = parser.parse_args()
    results = []
    for directory in args.logged:
        for path in sorted(Path(directory).glob("*.npz")):
            d = np.load(path)
            episode = speed_fit.load_one(path)
            steps, used = propagate_scale(episode, d["tracks"], d["K"], args.sigma_px, process=args.process,
                                          motion="rel_g" if args.oracle_motion else "rel_e")
            speed = steps / episode["dt"]
            truth = episode["speed"]
            flying = np.arange(len(truth)) >= 100
            norms = np.linalg.norm(episode["rel_e"][:, :3, 3], axis=1)
            results.append({
                "episode": episode["name"],
                "constant": speed_fit.chain_score(episode, np.where(norms > 1e-9, 1.0, 0.0)),
                "propagated": speed_fit.chain_score(episode, steps),
                "true_length": speed_fit.chain_score(episode, truth * episode["dt"]),
                "r": float(np.corrcoef(speed[flying], truth[flying])[0, 1]),
                "log_r": float(np.corrcoef(np.log(speed[flying] + 1e-3), np.log(truth[flying] + 1e-3))[0, 1]),
                "scale_drift": float(np.log(np.median(speed[flying][-200:] / np.maximum(truth[flying][-200:], 1e-3))
                                            / np.median(speed[flying][:200] / np.maximum(truth[flying][:200], 1e-3)))),
                "frames_measured": float(np.mean(used[flying] > 0)),
            })
    keys = ("constant", "propagated", "true_length", "r", "log_r", "scale_drift", "frames_measured")
    print(f"sigma_px {args.sigma_px}, process {args.process}\n{'episode':9s}" + "".join(f"{k:>16s}" for k in keys))
    for r in results:
        print(f"{r['episode']:9s}" + "".join(f"{r[k]:16.3f}" for k in keys))
    print(f"{'mean':9s}" + "".join(f"{np.mean([r[k] for r in results]):16.3f}" for k in keys))


if __name__ == "__main__":
    main()
