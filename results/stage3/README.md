# Stage 3 preparation (November 26, 2026)

Work on branch `victor/stage3`. Numbers are local (WASM sandbox unless noted) except the server section below.

## What changed for stage 3

The TA announcement: same trajectory distribution as stage 2 (agile flights that start in motion, unaligned ATE),
horizontal FoV uniform in 75–135° per episode, motion blur with exposure uniform in 0–50 ms, 12 of 24 episodes in rooms,
6 in a large empty warehouse, 6 in a forest. More (real-world) episodes may be added.

Our stage 2 estimator (`native/klt_bounded`, `r4_clamp`) cannot meet that: it assumes a start at rest for gravity, so on
in-flight starts its rotation error is 68–108° (cropped public flights, below), and its step length is a guessed speed.

## New estimator: `native/vio_ekf`

A compact MSCKF-style visual-inertial filter in C++ without OpenCV (about 1,500 lines, `solution.cpp` + `params.h`):

- State: attitude, position, velocity, gyro and accelerometer bias, **gravity vector** (world = IMU frame at frame 0, so
  in-flight starts need no level or rest assumption and the output needs no alignment), 3–4 cloned poses, optional
  SLAM points (world xyz).
- Front end: patches read straight from the RGB **green channel** (no gray image), integer-grid templates, IMU-predicted
  translation-only LK, Shi-Tomasi corners searched one or two grid cells per frame.
- Updates: features seen in >= 3 clones are triangulated (anchored inverse depth), projected onto the left null space of
  the point Jacobian (Mourikis & Roumeliotis 2007), chi-square gated, QR-compressed over the clone columns and applied in
  one EKF update. Low-parallax points keep an inverse-depth prior; a weak zero-mean velocity pseudo-measurement bounds
  IMU-only drift on blank walls; a watchdog restarts the filter from the last good pose on a non-finite state.
- Fuel: the update's rows are built on one frame and applied on the next (worst frame roughly halved), the corner search
  and velocity prior run on the frames without an update, f64x2/f32x4 SIMD kernels, sparse transition matrix, float32
  covariance option, per-update caps on features, candidates and new SLAM points.

Ideas taken from the class's stage 2 presentations (`https://docs.google.com/presentation/d/1tKtsiXloznSCBHEXUKXklbLLHL0Vh6cfRV13j7qyrhs`):

| Idea | From | Used as |
|---|---|---|
| EKF: IMU predicts, small image patches correct; <= 24 patches and 5 saved poses; green-channel sparse reads; SIMD; per-frame work caps | Team 3 (goats of 116, 0.276 m at 406k) | the overall design of `vio_ekf` |
| MSCKF with clones; gravity estimated for in-flight starts; velocity prior against runaways when tracks starve | Team 7 (Quadrotor lovers), Team 4, Team 8 | gravity in the state, `P_VPRIOR_SIGMA` |
| Level the expensive frames across a cycle | Team 5 (Team Robot), Team 9 | split update (`P_SPLIT`), balanced frames (`P_BALANCE`) |
| Feature carry: a corner keeps its depth after its keyframe leaves the window | Team 5 | SLAM points (`P_SLAM`) |
| IMU noise model must match the simulated sensor (gyro ~30x, accel ~10x EuRoC) | Team 2 (Bayesically Lost) | noise densities measured from a public hover |
| Watchdog that recovers from filter blow-ups; budget governor | Team 8 (Avengers) | `P_WATCHDOG`, update caps |
| Every stage 2 flight starts in motion; generate stage-2-like training/test flights | Teams 4, 5, 8 | `scripts/make_stage3_episodes.py` |

Not taken (different strategies, would be new work): learned velocity networks (Teams 4, 7, 8, 10), sliding-window
optimization with marginalization (Teams 5, 9), structural lines (Team 8), a probe submission that reads hidden-flight
statistics from the server's error report (Team 8; it costs an upload).

## Local data

`scripts/make_stage3_episodes.py` (+ `.sh`) simulates stage-3-like episodes with the pinned simulator: faster autopilot
(`scripts/stage3_autopilot.patch`, applied only while generating), the first 5 s cropped so the flight starts in motion,
FoV 105 ± 30°, and motion blur from averaging the 200 Hz renders over an exposure U(0, 50 ms) in linear light (the
simulator's camera hook). The public scene catalogs contain only ProcTHOR and HSSD houses, so there are no warehouse or
forest episodes; as a stand-in for larger scenes, `scripts/host_native.py` can scale any episode in space (`path@start:dur:k`:
positions and true accelerations x k, images unchanged), which is exactly the same flight in a k-times larger scene.

## Server results (Nov 26, 2026; `server.json`)

| Submission | Mean ATE | Peak fuel | On the frontier |
|---|---:|---:|---|
| `ekf_main` | 3.08 m | 1.1 M | no (goats: 1.5 m at 419k) |
| `ekf_ultra` | 4.0 m | 460k | no |
| `m_d` = `ekf_sliced` | 4.76 m | 230k | **yes** |

Other frontier points that day: Bayesically Lost 8.79 m at 81 fuel, `cpp_minimal` 7.55 m at 113 (the zero-output error of
the set is about 7.55 m), goats of 116 1.5 m at 419k, Bayesed Bots 0.7 m at 1.7 M, Team Robot 0.6 m at 4.1 M.

The server measures area in natural-log coordinates on both axes, so the open region is 113-419k fuel, where nothing
beats 7.55 m: `m_d` claims about ln(419/230) x ln(7.55/4.76) = 0.28. For comparison, `ekf_main` would need < 1.5 m to
claim anything, and about 1.2 m would claim only about 0.1. Hypothetical points: 100k at 4 m about 0.9, 20k at 4 m about 1.9.
`ekf_main`'s per-episode errors suggest the order rooms (12, mean 1.58 m), then 12 large-scene episodes (mean 4.6 m).

## Low-fuel EKF (`ekf_sliced`, Nov 26 afternoon)

`ekf_ultra` with 2 clones (`P_CLONES=2 P_MIN_OBS=2`) and these options (all off by default; `ekf_main` is unchanged):

- `P_DET_CAND`, `P_DET_REFINE`: sparse corner search (jittered candidate grid scored with a 5 x 5 Shi-Tomasi, hill
  climb). The dense search cost about 50-160k per frame.
- `P_FAST=2`: structured covariance propagation (scalar and skew blocks of the transition matrix, second-order bias
  blocks dropped), half-triangle velocity-prior and EKF covariance updates, branch-free clone removal (35k -> 9k),
  direct 2 x 2 green reads for windows and corner scores.
- `P_COV_DEFER`: covariance propagated only at clone frames over the accumulated interval (row building only reads the
  clone block, which propagation does not change; the velocity prior uses a one-frame-old covariance).
- `P_MEAN_FRAME`: mean propagated once per frame from interval sums. `P_MAX_CAND`: candidates per update.
- Also available: `P_VPRIOR_EVERY` (velocity prior on every k-th light frame), `P_POINT` (single green sample per
  pixel: cheaper but much less accurate from aliasing).

| Local | zero | `ekf_c2` (= submitted-day `e_c2`) | `ekf_sliced` |
|---|---:|---:|---:|
| s3 (16 x 45 s, sandbox) | 2.24 m | 1.88 m | 2.06 m |
| s3test (4 x 8 s, sandbox) | 2.23 m | 1.19 m | 0.93 m |
| agile proxy (4, sandbox) | 1.89 m | 0.59 m | 0.30 m |
| held-out scenes 88/150 (host) | 2.09 m | 1.84 m | 2.29 m |
| tune scenes x3 scale (host) | 7.16 m | 6.20 m | 5.64 m |
| held-out scenes x3 scale (host) | ~6.3 m | 5.53 m | 6.10 m |
| peak fuel (sandbox) | | 0.29 M | 0.20 M |

Server/local fuel ratio: about 1.1 (`ekf_ultra` 0.42 -> 0.46 M, `ekf_sliced` 0.20 -> 0.23 M).

Where the fuel goes now (s3 flights, by end-to-end removal): about 70k with no features (covariance about 30k on clone
frames, velocity prior about 25k on light frames, mean, clone handling, detection), plus about 12-15k per tracked feature
per frame (window, LK iterations at about 1k each, its share of the update). 8 features therefore cost about 100k; 6
features already lose accuracy (s3 2.48 m). Reaching about 100k needs fewer, cheaper features or a different estimator.
Note: the `P_DOUBLE` attribution (run a section twice) over-counts by about 2x in these builds; trust removal tests.
Small changes (even the first-frame corner search) move individual flights by up to about 1 m: compare on all sets.

## Cheap filter attempt (`native/vio_lite`, parked)

A 12-state filter (velocity, gravity, gyro bias, attitude error; no clones) whose visual input is the coplanarity of
feature bearings over a baseline plus a signed flow-magnitude row with an inverse-depth prior. Fixed along the way: a
sign ambiguity of the coplanarity (the signed flow row adds cheirality), attitude corrections not applied to the stored
history (`P_HISTCORR`), dense corner search cost. Measurement model verified with forced ground truth
(`scripts/host_lite_truth.py`, NIS < 1). Not robust: a fixed depth prior per feature is wrong for most features
(rooms span 0.5-5 m) and pulls the gyro bias (11-30 deg rotation error), so it is worse than zero output on the
blank-wall held-out scenes (3.3 m vs 2.1 m) at 75k median / 155k peak with 8 features. It would need per-feature depth
(what makes the EKF robust) and time-slicing to reach its intended 10-30k.

## Local results (WASM sandbox, default limits, mean unaligned ATE)

| Build | new stage-3-like (16 x 45 s) | stage-3-like test (4 x 8 s) | agile proxy, in-motion (4 x 7.5 s) | public, cropped in motion (10 x 30 s; deleted after the run, recreate in memory with `<hash>@15:30` specs) | agile proxy x2.5 scale (4) | public (10 x 60 s, hover start) | Peak fuel |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero output | 2.24 m | 2.23 m | 1.89 m | 4.55 m | 4.72 m | 4.89 m | |
| `ekf_main` | 1.31 m | 0.76 m | 0.37 m | 0.73 m | 3.13 m | 1.60 m | 0.95 M |
| `ekf_lite` | 1.91 m | 0.78 m | 0.42 m | 0.73 m | 3.66 m | 1.77 m | 0.59 M |
| `ekf_ultra` | 1.56 m | 0.90 m | 0.58 m | 1.05 m | 3.38 m | 1.71 m | 0.44 M |
| `ekf_c3` | 1.83 m | 1.08 m | 0.31 m | 0.79 m | 3.77 m | 1.57 m | 0.96 M |
| `ekf_slam8b` | 3.59 m | 1.03 m | 0.27 m | 0.92 m | 1.83 m | 1.69 m | 1.39 M |
| `r4_clamp` | 6.68 m | 1.15 m | 1.00 m | 5.08 m | 2.15 m | 5.72 m | 2.50 M |

Builds (`bash scripts/build_stage3.sh`; the record with per-episode numbers and wasm hashes is `results/stage3/local.json`):

- `ekf_main`: 3 clones, up to 20 features, priors tuned on the new set's scenes 12 and 45 (velocity prior 3 m/s per frame,
  inverse-depth prior std 0.2 at 3 m, initial gyro bias std 0.01 rad/s, bias walk 1e-4). On the held-out scenes 88 and 150
  it scores 1.68 m (host) against 2.29 m before tuning and 2.09 m for zero output; those scenes are mostly blank walls.
- `ekf_lite` / `ekf_ultra`: 12 / 8 features, 3 / 2 features per update, 5 x 5 patches; same priors.
- `ekf_c3`: `ekf_main` before the prior tuning. `ekf_slam8b`: with 8 SLAM points; better on the scaled agile set, much
  worse on the new set (landmarks lock in the scale they are born with).
- `r4_clamp`: our stage 2 submission (4.13 m on the stage 2 server).

The generated flights average 0.3-0.65 m/s (ProcTHOR rooms limit the faster autopilot), slower than the 0.7-1.3 m/s
other teams measured for stage 2, so they understate the agility; the warehouse and forest stand-in is only the scaled set.

## What limits accuracy

1. Heading drift from a wrongly learned gyro bias on blank walls (20-140 deg over 45 s on the worst scenes); the
   tighter bias prior halves it on average but it is still the largest error.
2. Metric scale: with 0.15-0.3 s clone windows the accelerometer constrains scale weakly; the depth and velocity priors
   provide it indoors and are wrong for larger scenes (scaled sets). SLAM points and a linear visual-inertial
   initialization (`P_VI_INIT`) were tried and did not fix this on the realistic data.
3. 10k fuel: covariance filtering alone costs far more (propagation + tracking ~0.3 M per frame at 3 clones), so the
   1 m at 10k goal needs a different estimator class, not tuning.

## Next steps

- The large area is below 419k fuel: cut `ekf_sliced` further (per-feature tracking cost, fewer but better features,
  velocity prior cost) or finish a much cheaper estimator; every halving of fuel at 4.76 m adds about 0.32.
- Large scenes are half the server set and every build is near zero output on the scaled stand-in; goats' 1.5 m
  suggests they handle them. Suspects: the 3 m inverse-depth prior and the zero-mean velocity prior.
- Long-baseline rotation constraint against heading drift; faster flights in larger HSSD scenes as a warehouse stand-in.

