"""Fast host loop for native solutions: compile solution.cpp as a shared library, drive it with ctypes over episodes
(the same inputs the sandbox gives), score with the upstream metrics. No fuel: use viobench evaluate for that.

    .venv-vio/bin/python scripts/host_native.py native/vio_ekf -D P_CLONES=6 --episodes work/data/s3/*.zip --jobs 8
"""
import argparse, ctypes, hashlib, json, os, subprocess, sys, glob
from multiprocessing import Pool
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "viobench/viobench/solution/runtime/native"


def build(src, defines, out_dir=ROOT / "work/hostlib"):
    out_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1((Path(src, "solution.cpp").read_text() + Path(src, "params.h").read_text() + " ".join(defines)).encode()).hexdigest()[:12]
    lib = out_dir / f"{Path(src).name}-{key}.so"
    if not lib.exists():
        cmd = ["g++", "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{RUNTIME}", f"-I{src}", *[f"-D{d}" for d in defines],
               str(Path(src, "solution.cpp")), "-o", str(lib)]
        subprocess.run(cmd, check=True)
    return lib


def virtual_episode(spec):
    """path[@start:duration[:scale]]: an episode cropped to start in motion and optionally scaled in space
    (positions and true accelerations x scale, images unchanged: the same flight in a larger scene, flown faster)."""
    from viobench.tools.datasets.sources import episodes
    clip = float(os.environ.get("GYRO_CLIP", "0"))
    if "@" not in spec:
        e = next(episodes(spec))
        if clip > 0:
            e.imu[:, 4:7] = np.clip(e.imu[:, 4:7], -clip, clip)
        return e
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_stage3_episodes import crop
    from viobench.tools.datasets.episode import Episode, FrameSequence
    path, rest = spec.split("@", 1)
    parts = [float(x) for x in rest.split(":")]
    start, dur = parts[0], parts[1]
    k = parts[2] if len(parts) > 2 else 1.0
    e = next(episodes(path))
    if start > 0:
        e = crop(e, start)
    n = int(np.searchsorted(e.frame_times, e.frame_times[0] + dur))
    fr = e.frames
    seq = FrameSequence(lambda i, _f=fr: _f[i], n, fr.frame_shape)
    t1 = e.frame_times[n - 1]
    imu = e.imu[e.imu[:, 0] <= t1 + 1e-9].copy()
    gt = e.ground_truth[e.ground_truth[:, 0] <= t1 + 0.05].copy()
    T = e.T_body_camera.copy()
    if k != 1.0:
        t = gt[:, 0]; pos = gt[:, 1:4]
        acc = np.gradient(np.gradient(pos, t, axis=0), t, axis=0)
        x, y, z, w = gt[:, 4:8].T
        R = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1), np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1),
                      np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
        ai = np.stack([np.interp(imu[:, 0], t, acc[:, j]) for j in range(3)], 1)
        idx = np.clip(np.searchsorted(t, imu[:, 0]), 0, len(t) - 1)
        imu[:, 1:4] += np.einsum("nji,nj->ni", R[idx], (k - 1) * ai)
        gt[:, 1:4] = pos[0] + k * (pos - pos[0])
        T[:3, 3] *= k
    if clip > 0:
        imu[:, 4:7] = np.clip(imu[:, 4:7], -clip, clip)
    cfg = {**e.config, "episode": {**e.config["episode"], "num_frames": n}}
    return Episode(name=e.name, config=cfg, frame_times=e.frame_times[:n], frames=seq, imu=imu, ground_truth=gt, T_body_camera=T, meta=e.meta)


def run_episode(args):
    lib_path, source = args
    from viobench.tools.runner.metrics import evaluate_trajectory
    lib = ctypes.CDLL(str(lib_path))
    lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    episode = virtual_episode(source)
    config, frames = episode.adapt((320, 240))
    blob = json.dumps(config).encode()
    if lib.init(blob, len(blob)) != 0:
        raise RuntimeError("init failed")
    est = np.zeros((episode.num_frames, 13))
    out = np.zeros(13)
    oracle_v = os.environ.get("ORACLE_V")
    for i, (t, frame, imu) in enumerate(episode.steps(frames)):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        imu = np.ascontiguousarray(imu, dtype=np.float64)
        if lib.step(float(t), frame.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data) != 0:
            raise RuntimeError(f"step {i} failed")
        est[i] = out
        if oracle_v and i == 0:
            # true velocity of the IMU at frame 0 in the IMU-0 frame (camera twist rotated by T_imu_cam)
            gt = episode.ground_truth_state()
            T = np.array(config["T_cam_imu"]); R_ci = T[:3, :3]
            v_cam = gt[0, 7:10]  # camera velocity in camera-0 axes (frame 0: camera axes = camera-0 axes)
            v_imu = R_ci.T @ v_cam
            lib.debug_set_velocity.argtypes = [ctypes.c_void_p, ctypes.c_double]
            vv = np.ascontiguousarray(v_imu, dtype=np.float64)
            lib.debug_set_velocity(vv.ctypes.data, float(oracle_v))
    m = evaluate_trajectory(episode.ground_truth_state(), est, episode.frame_times)
    return str(source), m.ate_rmse, m.sim3_ate_rmse, m.sim3_scale, m.rotation_rmse, m.velocity_rmse, m.distance_travelled


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("-D", action="append", default=[])
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--name", default=None)
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args()
    lib = build(a.src, a.D)
    eps = []
    for e in a.episodes:
        if "@" in e and any(c in e.split("@")[0] for c in "*?"):
            base, suffix = e.split("@", 1)
            eps += [f"{x}@{suffix}" for x in sorted(glob.glob(base))]
        else:
            eps += sorted(glob.glob(e)) if any(c in e for c in "*?") else [e]
    with Pool(a.jobs) as pool:
        res = pool.map(run_episode, [(lib, e) for e in eps], chunksize=1)
    ate = np.array([r[1] for r in res])
    for r in res:
        if not a.quiet:
            print(f"  {Path(r[0]).name[:40]:40s} ATE {r[1]:7.3f}  sim3 {r[2]:6.3f}  scale {r[3]:5.2f}  rot {np.degrees(r[4]):5.1f} deg  vel {r[5]:5.2f}  dist {r[6]:5.1f}")
    name = a.name or " ".join(a.D) or "default"
    print(f"{name}: mean ATE {ate.mean():.3f}  median {np.median(ate):.3f}  sim3 {np.mean([r[2] for r in res]):.3f}  "
          f"vel {np.mean([r[5] for r in res]):.3f}  n={len(res)}", flush=True)


if __name__ == "__main__":
    main()
