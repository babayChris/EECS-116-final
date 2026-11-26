"""vio_lite measurement-model check: true rotation, velocity and gravity are forced every frame; prints the normalized
innovation squared (NIS) of the coplanarity rows (should average ~1 if the model and its noise are right).
    .venv-vio/bin/python scripts/host_lite_truth.py <episode spec> [-D P_X=v ...]"""
import argparse, ctypes, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_native import build, virtual_episode

def qmat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

p = argparse.ArgumentParser(); p.add_argument("ep"); p.add_argument("-D", action="append", default=[]); p.add_argument("--src", default="native/vio_lite")
p.add_argument("--mode", type=int, default=7)
a = p.parse_args()
lib = ctypes.CDLL(str(build(a.src, a.D)))
lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.debug_truth.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int32]; lib.debug_nis.argtypes = [ctypes.c_void_p]
e = virtual_episode(a.ep)
cfg, frames = e.adapt((320, 240))
blob = json.dumps(cfg).encode(); lib.init(blob, len(blob))
raw = e.ground_truth
t = raw[:, 0]; pos = raw[:, 1:4]
vel = np.gradient(pos, t, axis=0)
R0 = qmat(raw[0, 4:8])
out = np.zeros(13); est = []
for i, (tf, fr, imu) in enumerate(e.steps(frames)):
    j = min(np.searchsorted(t, tf), len(t) - 1)
    Rk = R0.T @ qmat(raw[j, 4:8]); vk = R0.T @ vel[j]; gk = R0.T @ np.array([0, 0, -9.81])
    if a.mode:
        A = [np.ascontiguousarray(x, dtype=np.float64) for x in (Rk.reshape(-1), vk, gk)]
        lib.debug_truth(*[x.ctypes.data for x in A], a.mode)
    fr = np.ascontiguousarray(fr, dtype=np.uint8); imu = np.ascontiguousarray(imu, dtype=np.float64)
    lib.step(float(tf), fr.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data)
    est.append(out.copy())
o = np.zeros(10); lib.debug_nis(o.ctypes.data)
print(f"NIS mean {o[0] / max(o[1], 1):.2f} over {o[1]:.0f} rows; histogram <0.5 <1 <2 <4 <9 <25 <100 >100:", (o[2:] / max(o[1], 1)).round(3))
from viobench.tools.runner.metrics import evaluate_trajectory
m = evaluate_trajectory(e.ground_truth_state(), np.array(est), e.frame_times)
print(f"mode {a.mode}: ATE {m.ate_rmse:.3f} sim3 {m.sim3_ate_rmse:.3f} rot {np.degrees(m.rotation_rmse):.1f} vel {m.velocity_rmse:.2f}")
