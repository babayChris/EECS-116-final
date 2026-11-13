# EECS 116 Final

Reproducible local experiments with [VIOBench](https://github.com/jonas-eschmann/viobench), focused on trajectory accuracy versus maximum per-frame WebAssembly fuel.

**Current status:** one unmodified KLT + IMU baseline has completed one public episode in host debugging mode. WASM evaluation, feature-budget variants, and any improvement beyond the shared Pareto frontier are **not yet demonstrated**.

## Code and attribution

- `viobench/` is a Git submodule pinned to upstream commit [`17303aa14ba14e16cb54d9291d684371e1d3b548`](https://github.com/jonas-eschmann/viobench/tree/17303aa14ba14e16cb54d9291d684371e1d3b548), the revision used for the recorded run.
- The actual KLT estimator is [`viobench/solution/examples/python/klt_imu/solution.py`](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/viobench/solution/examples/python/klt_imu/solution.py) in that upstream repository. It is unmodified; this project does not claim authorship of VIOBench or its baselines.
- `setup.sh` installs the recorded host dependencies in an isolated environment.
- `run_local_vio.sh` runs KLT + IMU or IMU-only dead reckoning, in host or WASM mode.
- `scripts/score_saved_run.py` verifies the committed trajectory metrics using the upstream scorer.
- `results/klt-imu-host-first-public/` contains the measured report, original console log, and saved pose/twist estimates.

VIOBench declares MIT licensing in its [upstream package metadata](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/pyproject.toml). Additional upstream native estimators have their own licensing; consult their documentation before incorporating them.

## Setup

The recorded host environment was macOS arm64 with Python **3.14.6**. These scripts target macOS/Linux and require Git and Python 3.14. WASM builds additionally require the platform C compiler and `make`; CMake is installed by the setup. Other platforms are not verified here.

```bash
git clone --recurse-submodules https://github.com/babayChris/EECS-116-final.git
cd EECS-116-final
bash setup.sh
```

If `python3` is not Python 3.14, select the interpreter with `VIO_PYTHON=/path/to/python3.14 bash setup.sh`. `uv` is used if already installed; otherwise setup uses pip. The submodule is checked out at the committed revision, not automatically updated to upstream main.

## Run locally

Start with one public episode:

```bash
bash run_local_vio.sh host klt_imu 1
bash run_local_vio.sh host dead_reckoning 1
```

Host mode is useful for debugging and trajectory accuracy. It uses the host's libraries, has no sandbox limits, and does **not** measure competition fuel.

For official sandbox measurements:

```bash
bash run_local_vio.sh wasm klt_imu 1
bash run_local_vio.sh wasm dead_reckoning 1
```

The WASM commands are provided for the next experiment and have not yet completed in this project. The first Python bundle downloads and compiles its WASM runtime; upstream estimates 10–15 minutes, with actual time and disk use depending on the machine. Evaluation does not require a GPU. New HyperDrone training-data simulation requires an NVIDIA GPU according to the upstream CLI.

Use `10` instead of `1` for the complete public set in the pinned revision. Each episode is downloaded only when reached, with upstream estimating roughly 200 MB per archive. Data is cached under `work/conta-cache/`; set `CONTA_CACHE` to reuse another cache. Full runtime builds need additional disk space.

New reports and trajectory videos go under `outputs/local-benchmark/`. Repeating the same baseline and mode overwrites those generated outputs. The committed reference results under `results/` are left intact. Neither datasets, environments, downloaded toolchains nor generated binaries/videos are committed.

## Recorded result

Recorded November 13, 2026. Episode: `0386e5c941b50fe9972ddabff84c414540fb8989` (first public episode), 320 × 240 RGB.

| Measurement | Unmodified KLT + IMU, host mode |
|---|---:|
| Completed frames | 1,200 / 1,200 |
| Filled/invalid frames | 0 |
| Sim(3)-aligned trajectory RMSE | 1.975974 m |
| Unaligned trajectory RMSE | 482.192065 m |
| One-second relative translation RMSE | 21.576273 m |
| Fitted alignment scale | 0.012157 |
| Median host step time (rounded console value) | 0.7 ms |
| Maximum WASM fuel per frame | Not measured |

Host step timing excludes frame loading and video rendering; it is not interchangeable with deterministic WASM fuel. The aligned error must be interpreted alongside the severe unaligned drift. This single episode does not establish generalization or a Pareto improvement. No baseline parameters were tuned and no model was trained for this run.

The machine-readable report records the upstream revision, estimator SHA-256, dependency versions, and original command. Verify its metrics without rerunning the estimator:

```bash
.venv-vio/bin/python scripts/score_saved_run.py
```

This downloads the reference episode if it is missing from the cache. To use a known archive, pass `--archive /path/to/archive`. It also reports the same episode's constant-pose sanity-check error, approximately **4.345154 m**, and checks whether the pinned estimator matches the recorded source hash. Saved estimates have nine significant digits, as written by the upstream exporter.

## Next experiment

1. Obtain the live shared baseline scores and match their episode selection and evaluation limits.
2. Measure the unmodified estimator in WASM across the public set.
3. Compare a small number of variants with capped features, tracking work, and robust-estimation iterations.
4. Separately test initialization or fusion changes to identify their accuracy contribution.
5. Retain configurations that add useful area beyond the existing baseline frontier.

The scoring axes are equal-episode mean Sim(3) ATE and **maximum** fuel across all attempted frames. Skipping an expensive update on some frames may lower average work without reducing that maximum. Matching the original KLT baseline alone is insufficient evidence of a frontier improvement. See the [upstream task](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/docs/TASK.md) and [student guide](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/docs/STUDENT_GUIDE.md).
