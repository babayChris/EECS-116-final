# Agent handoff — EECS 116 Final

Last updated: November 13, 2026 (America/Los_Angeles).

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

## Not completed or not known

- No WASM runtime build, `check`, or `evaluate` run has completed in this project.
- No measured fuel, full-public-set benchmark, measured dead-reckoning run, or modified-estimator experiment exists here yet.
- No training, learned weights, feature-budget variants, or custom estimator have been produced.
- No competition submission has been made.
- The live server URL, current baseline points, selected evaluation hashes, and current server limits have not been obtained. A live Pareto comparison is therefore unavailable.
- Do not infer a Pareto win from the one-episode host result, upstream illustrative fuel numbers, or a paper's hardware timings.

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
