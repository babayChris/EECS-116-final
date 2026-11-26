"""Back-end test: perfect (noisy) feature observations of random static points along the true trajectory, real IMU.
    .venv-vio/bin/python scripts/host_synth.py <episode spec> [noise_px_gray] [depth_scale] [P_X=v ...]"""
import ctypes, json, sys, os
import numpy as np
sys.path.insert(0, "scripts")
from host_native import build, virtual_episode
from viobench.tools.runner.metrics import evaluate_trajectory
spec = sys.argv[1]; noise = float(sys.argv[2]); dscale = float(sys.argv[3]); defines = sys.argv[4:]
ds = 2
for d in defines:
    if d.startswith("P_DS="): ds = int(d.split("=")[1])
lib = ctypes.CDLL(str(build("native/vio_ekf", defines)))
lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.debug_set_features.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p]
e = virtual_episode(spec)
import os
if os.environ.get("PERFECT_IMU"):
    gt = e.ground_truth; t = gt[:, 0]; pos = gt[:, 1:4]
    def qms(q):
        x, y, z, w = q.T
        return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], -1), np.stack([2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], -1), np.stack([2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1)], -2)
    Rg = qms(gt[:, 4:8])
    vel = np.gradient(pos, t, axis=0); acc = np.gradient(vel, t, axis=0)
    imu = e.imu.copy()
    for j, ti in enumerate(imu[:, 0]):
        k = int(np.clip(np.searchsorted(t, ti), 1, len(t) - 1))  # sample at the end of [t_{k-1}, t_k]
        # mean specific force and rate over the interval [t_{k-1}, t_k]
        a_mid = (pos[k] - 2 * pos[k - 1] + pos[max(k - 2, 0)]) / (t[k] - t[k - 1]) ** 2 if k >= 2 else acc[k]
        Rm = Rg[k - 1]
        imu[j, 1:4] = Rm.T @ (0.5 * (acc[k] + acc[k - 1]) - np.array([0, 0, -9.81]))
        dR = Rg[k - 1].T @ Rg[k]
        imu[j, 4:7] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2 / (t[k] - t[k - 1])
    e.imu = imu
config, frames = e.adapt((320, 240))
fx, fy = config["fx"] / ds, config["fy"] / ds
cx, cy = (config["cx"] + 0.5) / ds - 0.5, (config["cy"] + 0.5) / ds - 0.5
W, H = 320 // ds, 240 // ds
cams = e.camera_poses_world()
def qm(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
rng = np.random.default_rng(0)
pts = {}; next_id = 0
blob = json.dumps(config).encode(); lib.init(blob, len(blob))
est = np.zeros((e.num_frames, 13)); out = np.zeros(13)
zero_frame = np.zeros((240, 320, 3), np.uint8)
for i, (t, _f, imu) in enumerate(e.steps(frames)):
    Rwc = qm(cams[i, 3:7]); pwc = cams[i, :3]
    vis = []
    for pid, P in list(pts.items()):
        Pc = Rwc.T @ (P - pwc)
        if Pc[2] > 0.2:
            u, v = fx * Pc[0] / Pc[2] + cx, fy * Pc[1] / Pc[2] + cy
            if 8 <= u < W - 8 and 8 <= v < H - 8:
                vis.append((pid, u + rng.normal(0, noise), v + rng.normal(0, noise))); continue
        del pts[pid]
    while len(vis) < 20:
        u, v = rng.uniform(10, W - 10), rng.uniform(10, H - 10)
        z = dscale * rng.uniform(1.0, 5.0)
        P = pwc + Rwc @ (np.array([(u - cx) / fx, (v - cy) / fy, 1.0]) * z)
        pts[next_id] = P; vis.append((next_id, u, v)); next_id += 1
    ids = np.array([p[0] for p in vis], np.int32); xy = np.array([[p[1], p[2]] for p in vis], np.float64).ravel()
    lib.debug_set_features(len(vis), ids.ctypes.data, xy.ctypes.data)
    imu = np.ascontiguousarray(imu)
    lib.step(float(t), zero_frame.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data)
    est[i] = out
    if os.environ.get("TRACE") and i % int(os.environ["TRACE"]) == 0:
        g = e.ground_truth_state()[i]
        dd = np.zeros(40); lib.debug_state.argtypes = [ctypes.c_void_p]; lib.debug_state(dd.ctypes.data)
        print(f"{i:4d} vi {dd[32]:.0f}/{dd[33]:.0f} sdv {dd[22]:.3f} used {dd[8]:.0f} pos err {np.linalg.norm(out[:3]-g[:3]):.3f}  v est {np.round(out[7:10],3)} gt {np.round(g[7:10],3)}  w est {np.round(out[10:13],3)} gt {np.round(g[10:13],3)}")
if os.environ.get("VI_DEBUG"):
    T = np.array(config["T_cam_imu"]); Rci = T[:3, :3]
    print("true v0 (IMU-0 frame):", np.round(Rci.T @ e.ground_truth_state()[0, 7:10], 3))
m = evaluate_trajectory(e.ground_truth_state(), est, e.frame_times)
print(f"{spec.split('/')[-1][:30]} noise {noise} depth x{dscale}: ATE {m.ate_rmse:.3f} sim3 {m.sim3_ate_rmse:.3f} scale {m.sim3_scale:.2f} rot {np.degrees(m.rotation_rmse):.1f} deg vel {m.velocity_rmse:.3f}")
