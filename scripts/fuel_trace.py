"""Per-frame sandbox fuel of one or more builds on one episode; with two builds, prints the per-frame difference.
    .venv-vio/bin/python scripts/fuel_trace.py <episode> a.wasm [b.wasm]"""
import sys
import numpy as np
from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.run import run_episode
from viobench.tools.runner.engine import Engine
import viobench.tools.runner.run as R
ep = next(episodes(sys.argv[1]))
engine = Engine()
limits = getattr(R, "DEFAULT_LIMITS", None)
fuels = []
for w in sys.argv[2:]:
    import tempfile; report, est, fuel = run_episode(w, ep, engine, limits, tempfile.mkdtemp())
    fuels.append(np.asarray(fuel))
    f = fuels[-1]
    print(f"{w}: max {f.max()/1e6:.3f} M at frame {int(f.argmax())}, median {np.median(f)/1e6:.3f} M, odd-frame median {np.median(f[1::2])/1e6:.3f}, even {np.median(f[2::2])/1e6:.3f}")
if len(fuels) == 2:
    d = fuels[1] - fuels[0]
    print(f"diff: median {np.median(d)/1e6:.3f} M, odd frames {np.median(d[1::2])/1e6:.3f}, even {np.median(d[2::2])/1e6:.3f}, max {d.max()/1e6:.3f} at {int(d.argmax())}")
    top = np.argsort(-fuels[0])[:5]
    print("top frames of the first build:", [(int(i), round(fuels[0][i] / 1e6, 3), round(d[i] / 1e6, 3)) for i in top])
