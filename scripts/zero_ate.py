"""Zero-output (never moving) unaligned ATE of episode specs (path[@start:dur:scale]), as host_native.py reads them.
    .venv-vio/bin/python scripts/zero_ate.py 'work/data/s3/*.zip@0:45:3' ..."""
import glob, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from host_native import virtual_episode
eps = []
for e in sys.argv[1:]:
    base, _, suf = e.partition("@")
    eps += [x + ("@" + suf if suf else "") for x in sorted(glob.glob(base))]
z = []
for s in eps:
    gt = virtual_episode(s).ground_truth_state()
    z.append(float(np.sqrt((gt[:, :3] ** 2).sum(1).mean())))
print(f"zero-output ATE mean {np.mean(z):.3f} over {len(z)}")
