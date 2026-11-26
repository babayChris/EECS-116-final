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
#define P_LK_ITERS 8
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
#define P_CLONES 5          // camera poses kept in the window
#endif
#ifndef P_CLONE_EVERY
#define P_CLONE_EVERY 2     // frames between clones
#endif
#ifndef P_MIN_OBS
#define P_MIN_OBS 3         // observations a feature needs for an update
#endif
#ifndef P_MAX_UPDATE_FEAT
#define P_MAX_UPDATE_FEAT 20
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
#define P_GYRO_WALK 2e-4    // rad/s^2/sqrt(Hz)
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
