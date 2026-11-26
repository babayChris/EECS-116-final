# EECS 116 Final

Team modulo's EECS 116/216 Project 3 code: a compute-aware visual-inertial odometry estimator for [VIOBench](https://github.com/jonas-eschmann/viobench), scored on trajectory error against the maximum WebAssembly fuel used by any frame (lower is better on both).

## Submissions

| Tag | Metric | Submitted build | Server result |
|---|---|---|---|
| `submission_stage1` | Mean Sim(3)-aligned ATE | Python `e5_p05_base3` (`solutions/klt_bounded/`), Nov 14 | 0.553 m at 20.1 M peak fuel (20 episodes) |
| `submission_stage2` | Mean unaligned ATE, private agile episodes | C++ `r4_clamp` and `r5_p0.42` (`native/klt_bounded/`), Nov 22 | 4.13 m at 2.6 M and 6.96 m at 1.3 M |

For reference on the stage 2 episodes, the stage 1 Python build scored 3.50 m at 20.1 M. Full server records are in `results/klt-bounded/server_nov14.json` and `results/klt-bounded/nov22_stage2.json`.

## Stage 3 work in progress (branch `victor/stage3`)

A new estimator, `native/vio_ekf`: a compact MSCKF-style visual-inertial EKF in C++ (no OpenCV) with gravity in the state,
so flights that start in motion need no calibration hover, and patch tracking read straight from the green channel.
`scripts/build_stage3.sh` builds the candidates; `scripts/make_stage3_episodes.sh` simulates stage-3-like flights (fast
autopilot, in-motion start, FoV 75–135°, motion blur 0–50 ms). Design, sources (including ideas from the class's stage 2
presentations) and local results: `results/stage3/README.md`. Nothing from it has been submitted.

## What the estimator does

It starts from the upstream KLT + IMU example and changes three things:

- **Bounded worst-frame work.** Corner caps, strip-wise replenishment, capped Lucas-Kanade iterations and robust-estimation limits keep every frame cheap, since the score uses the most expensive frame.
- **Gyro-aided motion.** The gyro gives the rotation between frames, so the translation direction needs only a 2-point fit instead of an essential matrix. Tracks persist across frames, and the gyro bias is corrected from vision.
- **Step length.** For stage 1, a constant speed (Sim(3) alignment fixes the scale). For stage 2, a small IMU-only speed model trained on the public episodes, with its output clamped to 0.5–1.5× its training mean because it extrapolated badly on agile flights.

The C++ port (stage 2) removes the Python runtime's 4.5 M per-frame floor and reaches about 2.5 M peak fuel at quarter resolution and 1.2 M at eighth resolution.

Local results on the 10 public episodes (WASM sandbox, default limits):

| Build | Unaligned ATE | Sim(3) ATE | Peak fuel/frame |
|---|---:|---:|---:|
| upstream `klt_imu` | — | 2.042 m | 142 M |
| `e5_p05_base3` (Python, stage 1) | — | 0.471 m | 19.8 M |
| `r4m` (C++, metric speed) | 1.017 m | 0.356 m | 2.49 M |
| `r5_p0.42` (C++, 1/8 resolution) | 1.010 m | 0.495 m | 1.22 M |

The public episodes are gentler than the stage 2 private set, which is why the server errors above are much larger. `REPORT.md` is the stage 1 write-up and predates both submissions. `AGENTS.md` has the dated log of every experiment, including negative results.

## Reproduce

After `bash setup.sh` (see Setup below) and one WASM runtime build (`bash run_local_vio.sh wasm klt_imu 1`):

```bash
bash scripts/build_stage2.sh                             # build the stage 2 submissions into work/cpp/*.wasm
bash scripts/eval_cpp.sh 10 4 r4_clamp r5_p0.42          # sandbox-evaluate them on the 10 public episodes
bash scripts/run_variants.sh 10 4 e5_p05_base3           # bundle + evaluate the stage 1 Python build
bash scripts/host_screen.sh 10 6 e5_p05_base3            # quick host accuracy screen (no fuel)
.venv-vio/bin/python tests/test_klt_bounded.py           # unit checks
```

`r4_clamp` rebuilds byte-identical to the submitted file. The submitted `r5_p0.42` was built before the clamp option existed; the current build differs only by two unused fields and computes the same output.

Other tools: `scripts/build_cpp_variant.sh` builds any C++ variant from parameter overrides, `scripts/make_heldout.sh` and `scripts/eval_heldout.sh` simulate and score a held-out set, and `scripts/make_agile_proxy.sh` with `scripts/eval_proxy.sh` generates and scores faster test flights (simulator setup in `AGENTS.md`).

## Code and attribution

- `viobench/` is a Git submodule pinned to upstream commit [`17303aa14ba14e16cb54d9291d684371e1d3b548`](https://github.com/jonas-eschmann/viobench/tree/17303aa14ba14e16cb54d9291d684371e1d3b548), the revision used for the recorded run.
- The upstream KLT estimator is [`viobench/solution/examples/python/klt_imu/solution.py`](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/viobench/solution/examples/python/klt_imu/solution.py) in that upstream repository. `solutions/klt_bounded/solution.py` is derived from it (attributed in the file), and `native/klt_bounded/` is our C++ port of that derived estimator. This project does not claim authorship of VIOBench or its baselines.
- `setup.sh` installs the recorded host dependencies in an isolated environment.
- `run_local_vio.sh` runs KLT + IMU or IMU-only dead reckoning, in host or WASM mode.
- `scripts/score_saved_run.py` verifies the committed trajectory metrics using the upstream scorer.
- `results/klt-imu-host-first-public/` contains the measured report, original console log, and saved pose/twist estimates.

VIOBench declares MIT licensing in its [upstream package metadata](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/pyproject.toml). Additional upstream native estimators have their own licensing; consult their documentation before incorporating them.

## Setup

The first host run used macOS arm64 with Python **3.14.6**; later experiments ran on Linux x86_64 (WSL2 and Arch) with wasmtime 50.0.1. These scripts target macOS/Linux and require Git and Python 3.14. WASM builds additionally require the platform C compiler and `make`; CMake is installed by the setup. Other platforms are not verified here.

```bash
git clone --recurse-submodules <this repository's URL>
cd <checkout directory>
bash setup.sh
```

If `python3` is not Python 3.14, select the interpreter with `VIO_PYTHON=/path/to/python3.14 bash setup.sh`. `uv` is used if already installed; otherwise setup uses pip. The submodule is checked out at the committed revision, not automatically updated to upstream main.

## Run the upstream baselines

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

The first Python bundle downloads and compiles its WASM runtime; upstream estimates 10–15 minutes, with actual time and disk use depending on the machine. Evaluation does not require a GPU. New HyperDrone training-data simulation requires an NVIDIA GPU according to the upstream CLI.

Use `10` instead of `1` for the complete public set in the pinned revision. Each episode is downloaded only when reached, with upstream estimating roughly 200 MB per archive. Data is cached under `work/conta-cache/`; set `CONTA_CACHE` to reuse another cache. Full runtime builds need additional disk space.

New reports and trajectory videos go under `outputs/local-benchmark/`. Repeating the same baseline and mode overwrites those generated outputs. The committed reference results under `results/` are left intact. Neither datasets, environments, downloaded toolchains nor generated binaries/videos are committed.

## First recorded baseline run

Recorded November 13, 2026, before any of our changes. Episode: `0386e5c941b50fe9972ddabff84c414540fb8989` (first public episode), 320 × 240 RGB.

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

## Known limitations

- Everything was tuned on the public episodes, which fly slower and more smoothly than the stage 2 private set. The fuel trims (20 corners, 5 px LK window, 5 iterations) were chosen on those gentler flights; on agile flights the trajectory shape (Sim(3) ATE 1.90 m) is worse than the slower Python stage 1 build (1.54 m).
- The learned speed model was trained on the public episodes and extrapolates poorly to faster flights; the clamp limits the damage rather than fixing it.
- One stage 2 episode (`726458d9bc31`) has about 0.7 rad of 1-second rotation error for every submission, ours and others', which suggests gyro saturation in the data.

The scoring axes are an equal-episode mean ATE and the **maximum** fuel across all attempted frames, so skipping an expensive update on some frames does not lower the score. See the [upstream task](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/docs/TASK.md) and [student guide](https://github.com/jonas-eschmann/viobench/blob/17303aa14ba14e16cb54d9291d684371e1d3b548/docs/STUDENT_GUIDE.md).
