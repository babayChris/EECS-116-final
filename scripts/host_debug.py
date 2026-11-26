"""Trace one episode through a native build (scripts/host_native.py) and print diagnostics every N frames."""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, "scripts")
from host_native import build

src, ep = sys.argv[1], sys.argv[2]
every = int(sys.argv[3]) if len(sys.argv) > 3 else 40
defines = sys.argv[4:]
lib = ctypes.CDLL(str(build(src, defines)))
lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.debug_state.argtypes = [ctypes.c_void_p]
from viobench.tools.datasets.sources import episodes
e = next(episodes(ep))
config, frames = e.adapt((320, 240))
print({k: config[k] for k in config if k != "T_cam_imu"}, config["T_cam_imu"])
blob = json.dumps(config).encode(); lib.init(blob, len(blob))
gt = e.ground_truth_state()
out = np.zeros(13); d = np.zeros(40)
for i, (t, frame, imu) in enumerate(e.steps(frames)):
    frame = np.ascontiguousarray(frame); imu = np.ascontiguousarray(imu)
    lib.step(float(t), frame.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data)
    import os
    if i == 0 and os.environ.get("ORACLE_V"):
        R_ci = np.array(config["T_cam_imu"])[:3, :3]; vv = np.ascontiguousarray(R_ci.T @ gt[0, 7:10])
        lib.debug_set_velocity.argtypes = [ctypes.c_void_p, ctypes.c_double]; lib.debug_set_velocity(vv.ctypes.data, float(os.environ["ORACLE_V"]))
    if i % every == 0 or i == e.num_frames - 1:
        lib.debug_state(d.ctypes.data)
        pe, pg = out[:3], gt[i, :3]
        print(f"{i:5d} feat {d[1]:3.0f} trk {d[3]:6.0f} lost {d[4]:5.0f} cand {d[5]:5.0f} trif {d[6]:5.0f} chif {d[7]:5.0f} used {d[8]:5.0f} | "
              f"pos err {np.linalg.norm(pe-pg):6.2f} |v| est {np.linalg.norm(out[7:10]):5.2f} gt {np.linalg.norm(gt[i,7:10]):5.2f} "
              f"slam {d[29]:.0f} init {d[30]:.0f} upd {d[31]:.0f} sdv {d[22]:.2f} bg {d[13]:+.3f} {d[14]:+.3f} {d[15]:+.3f} ba {d[16]:+.2f} {d[17]:+.2f} {d[18]:+.2f} |g| dir {d[19]:+.2f} {d[20]:+.2f} {d[21]:+.2f}")
