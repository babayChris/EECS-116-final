"""Per-section host cycle counts (rdtsc) of a native build compiled with -DP_PROF; prints the worst frames."""
import ctypes, json, sys
import numpy as np
sys.path.insert(0, "scripts")
from host_native import build
src, ep = sys.argv[1], sys.argv[2]
lib = ctypes.CDLL(str(build(src, ["P_PROF=1"] + sys.argv[3:])))
lib.init.argtypes = [ctypes.c_char_p, ctypes.c_uint32]
lib.step.argtypes = [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p]
lib.debug_prof.argtypes = [ctypes.c_void_p]
from viobench.tools.datasets.sources import episodes
e = next(episodes(ep)); config, frames = e.adapt((320, 240))
blob = json.dumps(config).encode(); lib.init(blob, len(blob))
out = np.zeros(13); d = np.zeros(8); rows = []
for i, (t, frame, imu) in enumerate(e.steps(frames)):
    frame = np.ascontiguousarray(frame); imu = np.ascontiguousarray(imu)
    lib.step(float(t), frame.ctypes.data, imu.ctypes.data, len(imu), out.ctypes.data)
    lib.debug_prof(d.ctypes.data); rows.append(d.copy())
rows = np.array(rows) / 1e3
tot = rows.sum(1)
names = ["prop", "track", "detect", "rows", "ekf", "tri", "chi2", "qr"]
print("median kcycles:", {n: round(float(np.median(rows[:, i])), 1) for i, n in enumerate(names)})
for i in np.argsort(-tot)[:6]:
    print(f"frame {i}: total {tot[i]:.0f} k  " + " ".join(f"{n} {rows[i, j]:.0f}" for j, n in enumerate(names)))
