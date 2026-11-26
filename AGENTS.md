# Agent handoff — EECS 116 Final

Last updated: November 26, 2026 (America/Los_Angeles), stage 3 preparation on `victor/stage3`. Sections below are dated; earlier ones are
snapshots and some of their "not yet" statements are superseded by later sections.

## Goal and conversation context

The user is planning a visual-inertial odometry project with VIOBench and wants a feasible approach that can extend its accuracy-versus-compute Pareto frontier. They first requested clarification of the first-stage deliverable, literature-backed approach ideas, a working Google Doc, local benchmarking, and then this public repository.

The user explicitly chose the repository name **EECS 116 final**, normalized to `babayChris/EECS-116-final`. Keep that name. A separate Google Doc and the original `Project 3.pdf` supplied context; neither is part of this repository. This handoff is sufficient for continuing the code work. Verify assignment-specific presentation requirements and deadlines against the handout before making claims about them.

The most recent technical recommendation was to investigate **bounded-compute KLT + IMU**, first reducing worst-frame visual work while preserving trajectory accuracy, then separately testing initialization/fusion improvements. This is a proposed direction, not a user-approved final algorithm or an implemented improvement. The user specifically asked whether it could beat the Pareto curve. The answer so far is: plausible, but unproven.

This file records context and next steps. Reading it does not itself request a new experiment, a server submission, or a deployment; follow the current user's task.

## Start here

1. Read `README.md`, inspect `git status`, and check `git submodule status`.
2. Read `results/klt-imu-host-first-public/host-results.json` and `run.log` for measured evidence.
3. Read `viobench/docs/TASK.md`, `STUDENT_GUIDE.md`, `DEBUGGING.md`, and `RUNTIME.md` at the pinned revision.
4. Inspect `run_local_vio.sh`, `setup.sh`, and `scripts/score_saved_run.py` before changing the workflow.

The upstream submodule is pinned to `17303aa14ba14e16cb54d9291d684371e1d3b548`. This is the revision used for the recorded results. Preserve the pin unless deliberately upgrading and documenting that change. The estimator is upstream code, not a custom implementation:

`viobench/viobench/solution/examples/python/klt_imu/solution.py`

It uses KLT tracks, an essential matrix for translation direction, gyro integration, and IMU-derived displacement scale. It detects up to 300 corners each frame and estimates initial gravity/gyro bias from approximately the first half second. It does not have a full filtering or bundle-adjustment backend or ongoing bias estimation. Do not describe it as an MSCKF or a complete optimized VIO system.

## Verified work completed

- Created and verified this public repository under `babayChris`.
- Installed dependencies in a separate `.venv-vio` and verified `setup.sh` in the packaged repository.
- Ran **unmodified KLT + IMU in host debugging mode** on the first public episode.
- Saved the original report, console log, 1,200 pose estimates, and 1,200 twist estimates under `results/klt-imu-host-first-public/`.
- Re-scored the saved estimates with the upstream scorer and reproduced all recorded metrics. Verified the estimator SHA-256 against the report.
- Added commands for both KLT and dead reckoning in host/WASM modes. Command availability does not mean those experiments have run.

Reference episode: `0386e5c941b50fe9972ddabff84c414540fb8989`, 320 × 240 RGB.

| Measurement | Recorded host run |
|---|---:|
| Completed frames | 1,200 / 1,200 |
| Filled/invalid frames | 0 |
| Sim(3)-aligned trajectory RMSE | 1.975974 m |
| Unaligned trajectory RMSE | 482.192065 m |
| One-second relative translation RMSE | 21.576273 m |
| Fitted alignment scale | 0.012157 |
| Median host step time, rounded log value | 0.7 ms |
| Maximum WASM fuel per frame | Not measured |

For a sanity check, scoring constant identity-pose predictions on the same episode gives **4.345154 m** aligned error. This is an offline calculation in `score_saved_run.py`, not a separately evaluated WASM baseline. The KLT result captures useful trajectory structure, but its large unaligned error indicates severe drift/scale problems. Sim(3) alignment can hide scale error.

## Update, November 13, 2026: WASM baselines and bounded-compute KLT (stage 1)

Measured on WSL2 Linux x86_64, Python 3.14.5, wasmtime 50.0.1, default local limits, all 10 pinned public episodes,
every run 10/10 completed with 0 filled frames. Compact record: `results/klt-bounded/summary.json`.

| | Mean Sim(3) ATE | Max fuel/frame | Median fuel |
|---|---:|---:|---:|
| minimal (upstream) | 3.239 m | 0.21 M | 0.09 M |
| dead_reckoning (upstream) | 2.152 m | 4.2 M | 4.1 M |
| klt_imu (upstream) | 2.042 m | 142.0 M | 58.2 M |
| `v23_quarter_c40` (ours) | 2.003 m | 35.7 M | 8.6 M |

- `solutions/klt_bounded/solution.py` is the upstream KLT estimator (attributed) with parameterized compute bounds;
  `variants.json` holds every measured configuration; `scripts/run_variants.sh` bundles and evaluates them.
- Diagnosis (`scripts/profile_klt.py`, `scripts/explain_fuel.py`): peak frames are essential-matrix RANSAC +
  recoverPose, whose cost scales with outliers via adaptive iterations; typical frames are LK cost per track. A one-time
  ~35 M cost on the first fully tracked frame is moved into init by a warm-up that drives a throwaway estimator copy.
- Local improvement with the server's `dominated_area` formula and the three upstream examples as stand-in baselines:
  +0.11% of the box. Not a server result. Accuracy, not fuel, is now the larger lever.
- `viobench simulate` (tiny_cnn training data) fails on WSL: OptiX unavailable; borrowed Linux 590.48 OptiX libraries
  initialize but module compilation fails (7299). Use native Windows or a Linux GPU machine.
- Progress page: `scripts/build_progress_page.py` writes `outputs/progress/index.html` from local reports.

## Update, November 13, 2026 (later): stage 2, accuracy at a bounded budget

Same setup, all 10 public episodes, every run completed with 0 filled frames; all in `results/klt-bounded/summary.json`.

| | Mean Sim(3) ATE | Max fuel/frame |
|---|---:|---:|
| klt_imu (upstream) | 2.042 m | 142.0 M |
| `v23_quarter_c40` (stage 1) | 2.003 m | 35.7 M |
| `v56_cs_par025_coast` | 0.637 m | 14.7 M |
| `v61_v56_h16` (best) | 0.562 m | 14.4 M |

- `direction: known_rotation`: gyro rotation between frames + 2-point linear hypotheses for the translation direction,
  epipolar-distance residuals, weighted refinement, cheirality sign. Median direction error vs ground truth 12° versus
  46–50° for the essential matrix (`scripts/direction_accuracy.py`). Unit checks: `tests/test_klt_bounded.py`.
- `fusion: constant_speed` + `no_vision: coast` + `min_derotated_parallax_px: 0.25`: the accelerometer no longer
  sets translation; Sim(3) alignment supplies the scale. This was the main accuracy gain. It assumes roughly steady
  flight speed and was tuned on the public validation episodes: check generalization before relying on it.
- Local improvement (server formula, stand-in baselines): +4.95% of the box. `scripts/host_screen.sh` screens accuracy
  on the host quickly; sandbox accuracy matched host accuracy exactly for all stage 2 variants.

## Update, November 13, 2026 (end of day): broader study, native baselines, held-out set

All sandbox numbers: WASM, default local limits, public set unless noted. Records: `results/klt-bounded/summary.json`
(sandbox), `host_study.json` (host screening, public + held-out), `heldout.json` (held-out set).

| | Public ATE | Held-out ATE (8 ep.) | Peak fuel/frame |
|---|---:|---:|---:|
| klt_imu (upstream) | 2.042 m | 1.574 m | 142 M (234 M held-out) |
| VINS-Mono (upstream native) | 0.604 m (1 episode failed) | not run | 1,591 M |
| OpenVINS (upstream native) | 0.420 m | not run | 3,548 M |
| `e3_persist05` | 0.520 m | 0.556 m | 14.2 M |
| `e4_p05_fb1` | 0.486 m | 0.521 m | 16.1 M |
| `e5_p05_base3` | 0.471 m | 0.513 m | 19.8 M |

- Evaluation environment: the server worker (`viobench/tools/server/worker.py`) runs the same `viobench evaluate`
  with its saved limits and the same pinned wasmtime 50.0.1; the upload contains our runtime, so results reproduce
  for the same episodes and limits. Unknown: the server's episode selection (`public+private-10` preset exists; the
  private episodes are not downloadable, and should not be sought) and limit changes.
- Held-out set: `scripts/make_heldout.sh` with `HYPERLUX_BACKEND=GENERIC` (CPU renderer, works under WSL; ProcTHOR
  ~40 min per one-minute episode, HSSD much slower and up to ~6.5 GB RAM). 8 of 10 episodes exist (hssd-b/c stopped).
  Never tune on it. `scripts/eval_heldout.sh` scores existing .wasm files on it.
- Study findings: seed noise ±0.02 m; persistent tracks (E3) clearly better and cheaper; forward-backward check (E4)
  and 3-frame baseline (E5) small gains; flow-based speed worse; tilt correction no effect; fuel floor without vision
  4.5 M (Python/IMU/state), so a native C++ port is the next fuel lever.
- Local improvement (server formula; stand-in baselines = 3 upstream Python examples + OpenVINS + VINS-Mono): +3.69%.
- Defined but not run: the `w_*` resolution × corner sweep on `c_best` (persist05 + fb1 + base3), and ORB-SLAM3's
  evaluation (it builds and passes check). The machine is shared with the user's other long jobs: limit parallelism.

## Update, November 14, 2026: server baselines, fuel sweep and metric scale (both negative)

Record: `results/klt-bounded/nov14_study.json`.

- Server (current set: 20 episodes; limits equal the local defaults: 20 G fuel/step, 200 G init, 2 GiB, 2 uploads/day).
  Baselines include `tiny_cnn` at 0.668 m / 14.7 M and VINS-Mono at 0.506 m / 1.88 G, so the estimated area for our
  points is about +0.9% (e5 alone) and +0.96% (e3 + e5), not the +3.69% against local stand-ins. Planned first
  uploads: `e5_p05_base3` and `e3_persist05`.
- Fuel sweep (`f*`, sandbox): more corners, resolution, pyramid levels or hypotheses (21-55 M) gave 0.465-0.495 m,
  within noise of e5. Vision quality is not the limit.
- Metric scale (`speed_model: "metric"`, `m*`, host): a sliding-window fit of speeds to IMU velocity changes along
  the visual directions. Every setting lost to constant speed (best 0.520 m). Cause: tilt error of 6-28 deg leaks
  1-4.7 m/s^2 of gravity, 10-50x the flights' true accelerations (median 0.09 m/s^2). The option stays in
  `solution.py`, off by default.
- Next: the native C++ port of the e3/e5 estimator, to remove the 4.5 M/frame Python floor (`native/klt_bounded/`).

## Update, November 15, 2026: C++ port, gyro bias from vision, low-fuel point

Record: `results/klt-bounded/nov15_cpp.json` (43 C++ variants, public and held-out). Build a variant with
`scripts/build_cpp_variant.sh <name> P_X=value ...` (copies `native/klt_bounded`, prepends the overrides to `params.h`),
evaluate with `scripts/eval_cpp.sh <episodes> <parallel> <names...>`.

| | Public ATE | Held-out ATE | Peak fuel |
|---|---:|---:|---:|
| Python `e5_p05_base3` | 0.471 m | 0.513 m | 19.8 M |
| C++ e5 (`c0_e5`) | 0.466 m | 0.512 m | 5.8 M |
| `n4_rf085_s3` (recommended main) | 0.405 m | 0.417 m | 3.0 M |
| `k7_c16_rf07` (recommended low-fuel) | 0.524 m | 0.612 m | 1.2 M |

Estimated server area for n4 + k7: +3.64% (public) / +3.39% (held-out numbers), against +0.96% for the Python pair.

- Gyro bias from vision (`P_BIAS_BASELINE=30`, Python `bias_baseline_frames`): the rotation correction that best
  explains tracks seen 30 frames apart moves the gyro bias (gain 0.02). True biases are 0.5-1.2 deg/s and the
  start-of-episode calibration misses them when the drone starts moving. Main accuracy gain of the day.
- Fuel: gray conversion dominated (2.4 M); a green-channel block mean (`P_FAST_GRAY=-1`) costs 0.7 M at 1/4 scale
  and 0.11 M at 1/8 scale with 2 rows x 4 columns (`P_GRAY_ROWS_8=2`). Peak frames were full-image corner
  detection on replenish (+1.4 M); `P_DETECT_STRIPS=3` bounds it but costs accuracy unless the replenish threshold
  is high (0.85).
- Oracle decomposition (host, b_B30_g0.02, public): true speed 0.410 -> 0.236 m, true direction -> 0.368, true
  rotation -> 0.404, speed + direction -> 0.018. Speed is the remaining error.
- Dead ends for speed: landmark triangulation (scale collapse; open-loop correlation with true speed 0.2-0.45),
  turn-rate model (speed correlates -0.4 with |gyro| but using it hurt), accelerometer (tilt leakage still > true
  acceleration with tilt correction). True drone tilt is about 0.5 deg: the flights are nearly level.

## Update, November 22, 2026: stage 2, unaligned ATE

Stage 2 (deadline Nov 22, 23:59) scores **unaligned ATE** (position RMSE in frame-0 axes; upstream viobench
`bd19c5f`) on 13 private "more agile" episodes (`private_stage2`, not downloadable, generator not published).
Sim(3) ATE is only a diagnostic now, so the step length must be in metres. Record:
`results/klt-bounded/nov22_stage2.json`; build every variant with `bash scripts/build_stage2.sh sweep`.

| Public set (sandbox) | Unaligned ATE | Sim(3) ATE | Peak fuel |
|---|---:|---:|---:|
| `r4` (speed rescaled to 0.5 m/s) | 2.354 m | 0.356 m | 2.49 M |
| `r4m` (`P_SPEED_PRIOR` = training mean 0.3246: metric speed) | 1.017 m | 0.356 m | 2.49 M |
| `n4m` (constant 0.3246 m/s, no speed model) | 1.148 m | 0.401 m | 2.48 M |
| `r5` (1/8 resolution, 0.5 m/s) | 1.315 m | 0.495 m | 1.22 M |
| `r5_p0.42` (1/8 resolution, 0.42 m/s) | 1.010 m | 0.495 m | 1.22 M |

- No code change: `learned_speed()` rescales the model's m/s prediction by `P_SPEED_PRIOR / MEAN`; setting the prior
  to the training mean makes the output metric.
- The 1/8-resolution build is best near 0.42 m/s (swept 0.3246-0.55), not at the true mean: noisier directions make
  the path zig-zag, so the net displacement is shorter than the distance stepped. At the true mean it scores 1.546 m.
- Submission candidates: `r5_p0.42` (dominates `r4m` on public) and `r4m` as a hedge, since the quarter-resolution
  build may track better on agile flights.
- Unknown: how fast the agile flights are. All tuning is on the older, gentler public flights; if the new flights are
  much faster, these speeds undershoot. The speed model was trained on old flights, so its extrapolation is untested.
- `victor/learned-speed` (Python flow + IMU speed, 26 M fuel) is superseded by the IMU-only C++ model; kept as the
  record of the flow and track-depth negative results.

### Stage 2 server results and the agile proxy (late Nov 22)

| Server, private_stage2 | Mean ATE | Sim(3) | Velocity RMSE | Peak fuel |
|---|---:|---:|---:|---:|
| `e5_p05_base3` (Python, Nov 14) | 3.50 m | 1.54 m | 0.87 m/s | 20.1 M |
| `r5_p0.42` | 6.96 m | 2.19 m | 1.87 m/s | 1.3 M |
| `r4_clamp` (`P_SPEED_MODE=1`, prior 0.75) | 4.13 m | 1.90 m | 0.80 m/s | 2.6 M |
| Christopher `eskf_e6_bg05` | 5.80 m | 2.14 m | 2.34 m/s | 1.3 M |

- The learned IMU speed model extrapolates badly on agile flights; clamping its ratio to the training mean to
  [0.5, 1.5] (`P_SPEED_MODE=1`, `P_SPEED_LO/HI`) cut ATE from 6.96 to 4.13 m and gave the lowest velocity error.
  `P_SPEED_MODE=2` (ratio to its own running mean) was worse on public and on the proxy.
- Remaining gap to `e5` is shape (Sim(3) 1.90 vs 1.54 m, drift/m about 2x): the fuel trims (20 corners, 5 px LK
  window, 5 iterations, strips) were tuned on gentle flights. Next: re-tune tracking on agile data.
- Episode `726458d9bc31` has about 0.7 rad 1-s rotation RPE for every submission: likely gyro saturation.
- Agile proxy: `scripts/agile_autopilot_proxy.patch` (apply to the pinned viobench, then `viobench simulate`;
  revert afterwards) raises the autopilot's speed, acceleration, jerk and yaw-rate limits. 10 s episodes reproduced
  the server's order, not its scale; use full-length episodes next time. Simulation needs `minizip` and
  `vulkan-headers` (Arch), `CPATH=/usr/include/minizip` and `LD_PRELOAD=/usr/lib/libcblas.so.3`; it renders at
  about 0.3 frames/s per episode, bound by one CPU thread. `scripts/eval_proxy.sh` scores builds on any episodes.

## Submissions and repositories (November 23, 2026)

- Stage 1: Python `e5_p05_base3`, submitted Nov 14 (`results/klt-bounded/server_nov14.json`). Stage 2: C++
  `r4_clamp` and `r5_p0.42`, submitted Nov 22 (`results/klt-bounded/nov22_stage2.json`).
- Tags `submission_stage1` (674a55a) and `submission_stage2` (`main` after merging `victor/stage2-metric`) exist in
  both the team repository and the course repository `ucb-eecs116/2026-fall-project3-modulo`, which instructors and
  other teams can read. Push only `main` and tags there; working branches stay in the team repository.
- Not merged into `main`: `qr/1115-screening`, `victor/learned-speed` (superseded), and Christopher's unpushed
  attempt (not to be merged, per Christopher).

## Update, November 26, 2026: stage 3 preparation (branch `victor/stage3`, nothing submitted)

Stage 3 (TA announcement): same trajectory distribution as stage 2, FoV U(75, 135) deg per episode, motion blur with
exposure U(0, 50) ms, 12/24 episodes in rooms, 6 in a large empty warehouse, 6 in a forest (`private_stage3`, 24 hashes in
upstream `viobench` `bc4de31`; the pin is still `17303aa`). Details, results and the list of ideas taken from the class's
stage 2 slides: `results/stage3/README.md`.

- New estimator `native/vio_ekf` (C++, no OpenCV): MSCKF-style EKF with gravity in the state (world = IMU frame at
  frame 0, so in-flight starts need no rest or level assumption), 3 clones, green-channel patch tracking, null-space
  updates, optional SLAM points. Every option is a `P_*` define in `native/vio_ekf/params.h`;
  `scripts/build_ekf_variant.sh <name> P_X=v ...` builds `work/ekf/<name>.wasm`; `scripts/build_stage3.sh` builds the
  candidates. Host loop (no fuel, ~10 s for 10 episodes): `scripts/host_native.py`, sweeps `scripts/ekf_sweep.sh`,
  traces `scripts/host_debug.py`, per-section host cycles `scripts/host_prof.py`, per-frame sandbox fuel
  `scripts/fuel_trace.py`, multi-set sandbox evaluation `scripts/eval_sets.py` (no videos).
- Why: `r4_clamp` has 68-108 deg rotation error on in-flight starts (its gravity comes from a start at rest), which is
  most of its 4.13 m stage 2 error. The EKF has 2-15 deg on the same flights.
- Episode generation: `scripts/make_stage3_episodes.sh <out> <seconds> "<scene>|<seed>" ...` (4 flights per job, faster
  autopilot patch applied only while running, first 5 s cropped, FoV and blur as announced). About 6 frames/min per job
  with 4 jobs; 16 x 45 s took about 2.7 h and about 1 GB. No warehouse or forest scenes exist in the public catalogs.
- `host_native.py` episode specs `path@start:duration:scale` crop and scale episodes in memory (scale k = same flight in
  a k-times larger scene), and `GYRO_CLIP=x` clips the gyro (saturation test).
- Findings: (1) metric scale from the IMU is weak with 0.15-0.3 s clone windows: even with perfect synthetic features
  (`scripts/host_synth.py`) the filter needs ~10 s of maneuvering to converge, and the first update linearised at v = 0
  locks a wrong velocity with high confidence. (2) The inverse-depth prior (3 m) and the zero-mean velocity prior are
  what make indoor flights work, but they make the filter underestimate motion in larger scenes (2.5x scaled flights).
  (3) SLAM points help the scaled flights and hurt some indoor ones. (4) A linear VI initialization (Martinelli-style,
  `P_VI_INIT`) is mathematically right (checked in Python) but too ill-conditioned over 1 s of gentle flight; off.
  (5) Gyro saturation detection (`P_CLIP_SIGMA`) recovers most of the loss from a clipped gyro; on by default.
- Fuel: WASM is a stack machine (about 7 instructions per multiply-add even with f64x2), so fuel follows operation
  counts: about 0.3 M per frame is propagation + tracking, the update adds 0.2-1 M with its rows x state size. 10k fuel
  is below the cost of any covariance filter here (a 36-state covariance update alone is tens of thousands).
- Not done: server submission, warehouse/forest data, learned components, the fuel goal of 10k.

STAGE3_RESULTS_PLACEHOLDER

## Not completed or not known

- The stage 2 private episodes and their generator are not available; all tuning used the gentler public flights.
- Held-out set incomplete (8 of 10; one HSSD episode).
- Do not infer a Pareto win from host results, upstream illustrative fuel numbers, or a paper's hardware timings.

## Scoring facts that determine the approach

- Lower is better on both axes: **equal-episode mean Sim(3)-aligned ATE** and **maximum fuel consumed by any attempted frame across the selected episodes**.
- Host wall-clock time is not Wasmtime fuel. Host NumPy/OpenCV also differ from their WASM builds; rerun in the sandbox before comparing submissions.
- Reducing average work or skipping expensive updates on some frames can leave maximum fuel unchanged. Bound the cost of the expensive frames themselves, including feature replenishment, RANSAC, and any initialization performed inside `step`.
- Adding a useful tradeoff to the frontier does not require beating every baseline on both axes. Matching an existing point adds no area; improving the supplied KLT alone may still leave the candidate dominated by another baseline.
- The server scores additional area beyond the baseline frontier in logarithmic coordinates. Use the implemented scorer and the actual shared baseline points rather than an invented success threshold.
- Failure does not remove the remaining frames from accuracy scoring: the current upstream rules fill missing states with origin/identity/zero twist. Inspect completion and filled-frame diagnostics alongside headline error.
- Inputs are causal RGB + IMU; camera intrinsics vary, and initial gravity alignment/rest cannot be assumed. Outputs are 13-value pose and twist states in the required frames/units.
- Match runtime version, episodes, and server limits. The checked-in defaults may differ from live settings. Public episodes are validation data, not a source of supervised training examples for a claimed generalization result.

## Recommended next technical work

When asked to continue benchmarking:

1. Verify disk space and reuse an existing episode cache when available. Build the pinned WASM runtime and run KLT on one episode as an initial check.
2. Measure unmodified KLT and dead reckoning over the pinned public set (10 episodes). Obtain the live baseline scores/settings when available; their absence need not block local measurements.
3. Implement a small, explicitly specified family of bounded-work variants: feature caps, bounded optical-flow/robust-estimation work, and limited feature replenishment. Example feature caps such as 80/150/300 are candidate settings, not measured recommendations.
4. Separately test gravity/bias initialization or unreliable-vision rejection changes. Keep comparisons controlled so changes in compute and accuracy can be attributed.
5. Keep configurations that extend the measured baseline frontier. If none do, reconsider the estimator before expanding the parameter search or starting a learning pipeline.

The upstream repository also includes OpenVINS, ORB-SLAM3, and VINS-Mono examples. Their existence does not establish that they are on the current live shared frontier. The tiny CNN example requires generated/trained `meta.json` and weights; these are absent from the pinned checkout. Generating HyperDrone training episodes requires NVIDIA GPU hardware according to the upstream CLI. Local CPU/WASM evaluation does not.

## Commands and local environment

The measured environment was macOS arm64, Python 3.14.6, Wasmtime 50.0.1, NumPy 2.5.4, and host OpenCV 5.0.0.93. `requirements-host.txt` pins the recorded packages. Python 3.14 is required by this repository's setup, even though upstream supports a broader range.

```bash
git submodule update --init --recursive
bash setup.sh
.venv-vio/bin/python scripts/score_saved_run.py
bash run_local_vio.sh host klt_imu 1
bash run_local_vio.sh wasm klt_imu 1
bash run_local_vio.sh wasm dead_reckoning 1
```

Use `VIO_PYTHON=/path/to/python3.14 bash setup.sh` if necessary. Use `10` instead of `1` for the full pinned public set. Upstream estimates 10–15 minutes for the first Python WASM build; this has not been verified on the user's Mac. Each public archive is roughly 200 MB. Check available space rather than assuming a full build or dataset fits.

`CONTA_CACHE` defaults to `work/conta-cache/`. On the original workstation, the first episode was cached in the parent workspace's `work/conta-cache/`, so `CONTA_CACHE="$PWD/../work/conta-cache"` can reuse it if that directory exists. Do not assume this layout on another checkout. Pass `--archive /path/to/archive` to the scorer to use an explicit archive.

The runner writes under `outputs/local-benchmark/` and overwrites generated outputs for the same mode/baseline. Preserve run-specific reports before rerunning. Committed reference results are separate under `results/`. The original trajectory video is local-only and is not required to reproduce metrics.

## Implementation and reporting conventions

- Keep the upstream baseline reproducible. Add custom solutions and their configuration in this repository with upstream attribution. Uncommitted changes inside a submodule are not included when the parent repository is pushed; do not leave the actual implementation only there.
- The existing helper accepts only `klt_imu` and `dead_reckoning`. Extend it deliberately or document a separate command when adding a variant.
- Record code revision, configuration, module/source hash, runtime/library versions, episode hashes, limits, completion, aligned/unaligned errors, and maximum fuel for new sandbox experiments. Keep representative reports small enough for Git.
- Preserve measured results and distinguish them from proposed work, upstream examples, and literature claims. Never populate missing fuel with host timing or illustrative upstream values.
- Keep dependencies, datasets, toolchains, build outputs, credentials, and private assignment/account documents out of this public repository. Existing ignore rules cover the common generated paths.
- No extra tests were added for documentation. Useful checks for workflow changes are shell syntax, a clean setup, and re-scoring the saved trajectory. A WASM-path change needs an actual sandbox check before calling that path verified.
- Give the user concise explanations of what ran, what changed, what remains unmeasured, and the next decision. Do not promise a Pareto improvement before measuring it.
- Update this handoff and the README when the state changes so a later agent does not mistake this snapshot for current results.

## Literature context from the earlier discussion

The prior discussion used a targeted literature scan, not a systematic review. These are leads supporting possible designs; no paper's gains have been reproduced here. Recheck the primary source before making precise claims or selecting an implementation.

- [HybVIO (WACV 2022)](https://arxiv.org/abs/2106.11857): feature selection/filtering as inspiration for bounded visual work. Do not confuse the number of selected update features with the total tracked-feature count.
- [LEVIO (2026)](https://arxiv.org/abs/2602.03294): embedded lightweight VIO. Hardware-specific speed/power results do not predict WASM fuel.
- [UL-VIO (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08054.pdf): a compact learned alternative; model size alone is not a compute guarantee.
- [Learned IMU Bias Prediction (2025)](https://arxiv.org/abs/2505.06748): a possible later bias-learning extension; compare it against conventional bias estimation and use separate training data.
- [Robust and Efficient Visual-Inertial Initialization (2025)](https://arxiv.org/abs/2410.19473): a lead for gravity, velocity, scale, and bias initialization.
- [Dual-Agent Adaptive and Cost-Aware VIO](https://arxiv.org/abs/2511.21083): scheduling motivation, but reducing visual-call frequency is different from minimizing VIOBench's peak-frame fuel.
- [LBDU-VIO (2026 preprint)](https://arxiv.org/abs/2609.39125): exploratory bias-dynamics lead; the earlier discussion inspected only its abstract, so detailed methodological claims are not established.
