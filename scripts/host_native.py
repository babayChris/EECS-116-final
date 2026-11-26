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


def run_episode(args):
    lib_path, source = args
    from viobench.tools.datasets.sources import episodes
    from viobench.tools.runner.metrics import evaluate_trajectory
    lib = ctypes.CDLL(str(lib_path))
    lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
    lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
    episode = next(episodes(source))
    config, frames = episode.adapt((320, 240))
    blob = json.dumps(config).encode()
    if lib.init(blob, len(blob)) != 0:
        raise RuntimeError("init failed")
    est = np.zeros((episode.num_frames, 13))
    out = np.zeros(13)
    for i, (t, frame, imu) in enumerate(episode.steps(frames)):
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        imu = np.ascontiguousarray(imu, dtype=np.float64)
        if lib.step(float(t), frame.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data) != 0:
            raise RuntimeError(f"step {i} failed")
        est[i] = out
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
