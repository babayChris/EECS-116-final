# Stage 3 preparation (November 26, 2026)

Work on branch `victor/stage3`. Nothing here has been submitted; all numbers are local (WASM sandbox unless noted).

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

RESULTS_PLACEHOLDER
