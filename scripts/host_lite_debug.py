"""Per-second trace of vio_lite against ground truth: speed, velocity error, gravity error, gyro bias.
    .venv-vio/bin/python scripts/host_lite_debug.py <episode spec> [-D P_X=v ...]"""
import argparse, ctypes, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_native import build, virtual_episode, ROOT

p = argparse.ArgumentParser(); p.add_argument("ep"); p.add_argument("-D", action="append", default=[]); p.add_argument("--src", default="native/vio_lite")
a = p.parse_args()
lib = ctypes.CDLL(str(build(a.src, a.D)))
lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.debug_state.argtypes = [ctypes.c_void_p]
e = virtual_episode(a.ep)
cfg, frames = e.adapt((320, 240))
blob = json.dumps(cfg).encode(); lib.init(blob, len(blob))
gt = e.ground_truth_state()
raw = e.ground_truth
x, y, z, w = raw[0, 4:8]
R0 = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
g_true = R0.T @ np.array([0, 0, -9.81])
out = np.zeros(13); st = np.zeros(32)
T = np.array(cfg["T_cam_imu"]); Rci = T[:3, :3]
for i, (t, fr, imu) in enumerate(e.steps(frames)):
    fr = np.ascontiguousarray(fr, dtype=np.uint8); imu = np.ascontiguousarray(imu, dtype=np.float64)
    lib.step(float(t), fr.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data)
    if i % 20 == 0:
        lib.debug_state(st.ctypes.data)
        ve, vt = out[7:10], gt[i, 7:10]
        g = st[7:10]
        gerr = np.degrees(np.arccos(np.clip(g @ g_true / np.linalg.norm(g) / 9.81, -1, 1)))
        perr = np.linalg.norm(out[:3] - gt[i, :3])
        qd = abs(float(np.dot(out[3:7], gt[i, 3:7]))); rerr = np.degrees(2 * np.arccos(min(1.0, qd)))
        print(f"t {t - e.frame_times[0]:5.1f} feats {st[1]:2.0f} used {st[2]:5.0f} rej {st[3]:4.0f} |v| est {np.linalg.norm(ve):4.2f} true {np.linalg.norm(vt):4.2f} "
              f"verr {np.linalg.norm(ve - vt):4.2f} sdv {st[13]:4.2f} gerr {gerr:4.1f}deg |g| {np.linalg.norm(g):5.2f} bg {st[10]:+.4f} {st[11]:+.4f} {st[12]:+.4f} rot {rerr:5.1f} perr {perr:5.2f} |p| {np.linalg.norm(gt[i,:3]):5.2f}")
