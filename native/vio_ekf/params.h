// Parameters of the vio_ekf estimator. Every value can be overridden by a #define before this file
// (scripts/build_cpp_variant.sh-style builds prepend them).
#pragma once

// ---- image front end
#ifndef P_DS
#define P_DS 2              // gray image = green channel, P_DS x P_DS block mean of the 320 x 240 frame
#endif
#ifndef P_MAX_FEAT
#define P_MAX_FEAT 30       // tracked features
#endif
#ifndef P_GRID_X
#define P_GRID_X 5
#endif
#ifndef P_GRID_Y
#define P_GRID_Y 4
#endif
#ifndef P_DETECT_BELOW
#define P_DETECT_BELOW 24   // detect new corners when fewer features are tracked
#endif
#ifndef P_DETECT_STEP
#define P_DETECT_STEP 2     // corner score evaluated every P_DETECT_STEP pixels
#endif
#ifndef P_MIN_SCORE
#define P_MIN_SCORE 20.0    // minimum Shi-Tomasi score (gray levels^2 per pixel)
#endif
#ifndef P_HALF
#define P_HALF 3            // LK patch half size: (2 P_HALF + 1)^2 pixels
#endif
#ifndef P_LK_ITERS
#define P_LK_ITERS 5
#endif
#ifndef P_LK_MAX_ERR
#define P_LK_MAX_ERR 18.0   // mean absolute photometric error for a track to survive
#endif
#ifndef P_PRED_DEPTH
#define P_PRED_DEPTH 2.0    // depth (m) assumed when predicting a feature's motion from the filter velocity
#endif
#ifndef P_FB_CHECK
#define P_FB_CHECK 0        // forward-backward check: 1 = on
#endif

// ---- filter
#ifndef P_CLONES
#define P_CLONES 4          // camera poses kept in the window
#endif
#ifndef P_CLONE_EVERY
#define P_CLONE_EVERY 2     // frames between clones
#endif
#ifndef P_MIN_OBS
#define P_MIN_OBS 3         // observations a feature needs for an update
#endif
#ifndef P_MAX_UPDATE_FEAT
#define P_MAX_UPDATE_FEAT 6
#endif
#ifndef P_SIGMA_PX
#define P_SIGMA_PX 1.0      // pixel noise at 320 x 240
#endif
#ifndef P_MAX_REPROJ_PX
#define P_MAX_REPROJ_PX 3.0 // triangulation reprojection RMS gate at 320 x 240
#endif
#ifndef P_CHI2
#define P_CHI2 1            // per-feature chi-square gate
#endif
#ifndef P_CHI2_SCALE
#define P_CHI2_SCALE 1.0
#endif
#ifndef P_GYRO_NOISE
#define P_GYRO_NOISE 0.008  // rad/s/sqrt(Hz)
#endif
#ifndef P_ACC_NOISE
#define P_ACC_NOISE 0.04    // m/s^2/sqrt(Hz)
#endif
#ifndef P_GYRO_WALK
#define P_GYRO_WALK 1e-3    // rad/s^2/sqrt(Hz)
#endif
#ifndef P_ACC_WALK
#define P_ACC_WALK 2e-3     // m/s^3/sqrt(Hz)
#endif
#ifndef P_INIT_TILT
#define P_INIT_TILT 0.3     // rad, std of the initial gravity direction
#endif
#ifndef P_INIT_VEL
#define P_INIT_VEL 1.5      // m/s, std of the initial velocity (episodes may start in motion)
#endif
#ifndef P_INIT_BG
#define P_INIT_BG 0.02      // rad/s
#endif
#ifndef P_INIT_BA
#define P_INIT_BA 0.2       // m/s^2
#endif
#ifndef P_MAX_SPEED
#define P_MAX_SPEED 6.0     // m/s: a velocity estimate above this is clamped (runaway guard)
#endif
#ifndef P_DEPTH_PRIOR
#define P_DEPTH_PRIOR 3.0   // m: prior mean depth of a feature (inverse-depth prior, see P_RHO_SIGMA)
#endif
#ifndef P_RHO_SIGMA
#define P_RHO_SIGMA 0.15    // 1/m: std of the inverse-depth prior (0 = no prior: project the whole point out)
#endif
#ifndef P_CARRY_DEPTH
#define P_CARRY_DEPTH 0     // a used feature keeps its triangulated point as the depth prior of its next track
#endif
#ifndef P_CARRY_INFLATE
#define P_CARRY_INFLATE 2.0 // the carried inverse-depth std is the posterior std times this
#endif
#ifndef P_RHO_SIGMA_MIN
#define P_RHO_SIGMA_MIN 0.02
#endif
#ifndef P_DETECT_CELLS
#define P_DETECT_CELLS 2    // empty grid cells searched for a new corner per frame
#endif
#ifndef P_SPLIT
#define P_SPLIT 1           // MSCKF rows at clone frames, the EKF update at the next frame (halves the peak)
#endif
#ifndef P_PRIOR_SWITCH
#define P_PRIOR_SWITCH 0.3  // no depth prior for a point whose triangulated inverse-depth std is below this fraction (0 = always prior)
#endif
#ifndef P_LK_REL_ERR
#define P_LK_REL_ERR 0.3    // a track is lost if its mean absolute error exceeds this times the patch contrast (+1)
#endif
#ifndef P_VPRIOR_SIGMA
#define P_VPRIOR_SIGMA 5    // m/s per frame: weak zero-mean velocity pseudo-measurement every frame (0 = off)
#endif
#ifndef P_SKIP
#define P_SKIP 0            // fuel attribution only: 1 = no tracking, 2 = no detection, 4 = no velocity prior, 8 = no propagation
#endif
#ifndef P_MARGIN
#define P_MARGIN 4          // search margin (gray pixels) of a tracking window around the predicted position
#endif
#ifndef P_DOUBLE
#define P_DOUBLE 0          // fuel attribution only: run sections twice on a copy (1 track, 2 detect, 4 vprior, 8 propagate, 16 ekf, 32 rows, 64 copy only)
#endif
#ifndef P_ADAPT_DEPTH
#define P_ADAPT_DEPTH 0     // depth prior from the running mean log-depth of well-triangulated points
#endif
#ifndef P_RHO_REL
#define P_RHO_REL 0.5       // adaptive prior: inverse-depth std relative to the prior inverse depth
#endif
#ifndef P_DEPTH_RATE
#define P_DEPTH_RATE 0.05   // adaptive prior: update rate of the running log-depth
#endif
#ifndef P_DETECT_CELLS_FIRST
#define P_DETECT_CELLS_FIRST 4  // cells searched on the first frame (the rest follow round-robin)
#endif
#ifndef P_BALANCE
#define P_BALANCE 1         // split mode: velocity prior and corner search only on frames without an EKF update
#endif
#ifndef P_TRI_ITERS
#define P_TRI_ITERS 8       // Gauss-Newton iterations of the triangulation
#endif
#ifndef P_CHI2_FAST
#define P_CHI2_FAST 0       // accept a feature without the chi-square test when its residual^2 < this * rows * sigma^2
#endif
#ifndef P_FLOATCOV
#define P_FLOATCOV 0        // covariance and EKF update in float32 (f32x4 SIMD)
#endif
#ifndef P_EKF_SKIP
#define P_EKF_SKIP 0        // fuel attribution only: skip EKF parts (1 PHt, 2 S, 4 W, 8 P update, 16 correction)
#endif
#ifndef P_BENCH
#define P_BENCH 0           // fuel micro-benchmark in init (1 PHt, 2 S, 3 P update, 4 W solve), 100 repetitions
#endif
#ifndef P_SLAM
#define P_SLAM 0            // persistent SLAM points in the state (world xyz)
#endif
#ifndef P_SLAM_REL
#define P_SLAM_REL 0.15     // a feature becomes a SLAM point if its inverse-depth std is below this fraction
#endif
#ifndef P_SLAM_EVERY
#define P_SLAM_EVERY 2      // SLAM point updates every this many frames
#endif
#ifndef P_SLAM_CHI2
#define P_SLAM_CHI2 9.21    // per-point chi-square gate (2 dof, 99 %)
#endif
#ifndef P_SLAM_ROWS
#define P_SLAM_ROWS 1       // a new SLAM point's remaining (null-space) rows also update the state
#endif
#ifndef P_VZ_SIGMA
#define P_VZ_SIGMA 0        // m/s per frame: vertical velocity pseudo-measurement (0 = off)
#endif
#ifndef P_SLAM_MAXUPD
#define P_SLAM_MAXUPD 8     // SLAM points per update (round-robin when there are more)
#endif
#ifndef P_WATCHDOG
#define P_WATCHDOG 1        // restart the filter from the last good pose on a non-finite or runaway state
#endif
#ifndef P_SLAM_VSIG
#define P_SLAM_VSIG 1e9     // SLAM points only once the velocity std is below this (m/s)
#endif
#ifndef P_VI_INIT
#define P_VI_INIT 0         // frames of linear visual-inertial initialization before the filter starts (0 = off)
#endif
#ifndef P_VI_MAX
#define P_VI_MAX 60         // give up the initialization after this many frames
#endif
#ifndef P_VI_RETRY
#define P_VI_RETRY 5        // frames between solve attempts after P_VI_INIT
#endif
#ifndef P_VI_MINFEAT
#define P_VI_MINFEAT 6      // points with >= 3 observations needed for a solve
#endif
#ifndef P_VI_MINSPAN
#define P_VI_MINSPAN 6      // frames a point must span to enter the solve
#endif
#ifndef P_VI_VSIG
#define P_VI_VSIG 0.2       // velocity std after the initialization
#endif
#ifndef P_VI_TILT
#define P_VI_TILT 0.05      // tilt std after the initialization (rad)
#endif
#ifndef P_MAX_CAND
#define P_MAX_CAND 100      // features tried per update (triangulation cost bound)
#endif
#ifndef P_SLAM_INIT_MAX
#define P_SLAM_INIT_MAX 100 // new SLAM points per update
#endif
#ifndef P_PHI_FRAME
#define P_PHI_FRAME 0       // transition matrix once per frame from interval sums instead of per IMU sample
#endif
#ifndef P_DETECT_MULT
#define P_DETECT_MULT P_CLONE_EVERY  // split mode: cells searched on a light frame = P_DETECT_CELLS * this
#endif
