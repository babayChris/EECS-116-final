"""Per-frame sandbox fuel summary of builds on one episode: max, p99, median, top frames, medians by frame mod 4.
    .venv-vio/bin/python scripts/fuel_frames.py <episode> a.wasm [b.wasm ...]"""
import sys, tempfile
import numpy as np
from viobench.tools.datasets.sources import episodes
from viobench.tools.runner.run import run_episode
from viobench.tools.runner.engine import Engine
ep = next(episodes(sys.argv[1]))
eng = Engine()
for w in sys.argv[2:]:
    _, _, f = run_episode(w, ep, eng, None, tempfile.mkdtemp()); f = np.asarray(f)
    print(f"{w.split('/')[-1]:14s} max {f.max()/1e3:4.0f}k p99 {np.percentile(f, 99)/1e3:4.0f}k median {np.median(f)/1e3:4.0f}k top",
          [(int(i), int(f[i] / 1e3)) for i in np.argsort(-f)[:5]], "by frame%4", [int(np.median(f[k::4]) / 1e3) for k in range(4)], flush=True)
