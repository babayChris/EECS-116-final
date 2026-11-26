// Parameters of the vio_lite estimator (low-fuel velocity filter). Override with #define before this file.
#pragma once
#ifndef P_DS
#define P_DS 2              // gray resolution = 320 x 240 / P_DS
#endif
#ifndef P_POINT
#define P_POINT 1           // 1 = gray pixel is one green sample (no block mean)
#endif
#ifndef P_MAX_FEAT
#define P_MAX_FEAT 8
#endif
#ifndef P_GRID_X
#define P_GRID_X 4
#endif
#ifndef P_GRID_Y
#define P_GRID_Y 3
#endif
#ifndef P_DETECT_STEP
#define P_DETECT_STEP 2
#endif
#ifndef P_DETECT_CELLS
#define P_DETECT_CELLS 1    // grid cells searched per frame while features are missing
#endif
#ifndef P_DET_CAND
#define P_DET_CAND 16      // random candidate points scored per searched cell (0 = dense search)
#endif
#ifndef P_MIN_SCORE
#define P_MIN_SCORE 20.0
#endif
#ifndef P_HALF
#define P_HALF 2            // LK patch (2 P_HALF + 1)^2
#endif
#ifndef P_MARGIN
#define P_MARGIN 3
#endif
#ifndef P_LK_ITERS
#define P_LK_ITERS 3
#endif
#ifndef P_LK_MAX_ERR
#define P_LK_MAX_ERR 18.0
#endif
#ifndef P_LK_REL_ERR
#define P_LK_REL_ERR 0.3
#endif
#ifndef P_BASE
#define P_BASE 6            // frames between the two bearings of a coplanarity measurement
#endif
#ifndef P_MIN_BASE
#define P_MIN_BASE 4        // shortest baseline used (frames)
#endif
#ifndef P_MEAS_EVERY
#define P_MEAS_EVERY 1      // a feature gives a measurement every this many frames (staggered by id)
#endif
#ifndef P_SIGMA_PX
#define P_SIGMA_PX 1.0      // pixel noise at the gray resolution
#endif
#ifndef P_MEAS_INFLATE
#define P_MEAS_INFLATE 1.0  // measurement variance multiplier (overlapping baselines are correlated)
#endif
#ifndef P_CHI2
#define P_CHI2 9.0
#endif
#ifndef P_ACC_NOISE
#define P_ACC_NOISE 0.3     // m/s^2/sqrt(Hz): accelerometer noise + model error (no accel bias state)
#endif
#ifndef P_G_WALK
#define P_G_WALK 0.02       // m/s^2/sqrt(s): gravity (in the drifting world frame) random walk
#endif
#ifndef P_GYRO_WALK
#define P_GYRO_WALK 1e-4
#endif
#ifndef P_INIT_BG
#define P_INIT_BG 0.01
#endif
#ifndef P_INIT_VEL
#define P_INIT_VEL 1.5
#endif
#ifndef P_INIT_TILT
#define P_INIT_TILT 0.3
#endif
#ifndef P_GNORM_SIGMA
#define P_GNORM_SIGMA 0.05  // |g| = 9.81 pseudo-measurement per frame (0 = off)
#endif
#ifndef P_VPRIOR_SIGMA
#define P_VPRIOR_SIGMA 0    // m/s per frame zero-mean velocity pseudo-measurement (0 = off)
#endif
#ifndef P_MAX_SPEED
#define P_MAX_SPEED 8.0
#endif
#ifndef P_FRAME_IMU
#define P_FRAME_IMU 1       // 1 = IMU integrated once per frame from interval sums
#endif
#ifndef P_BIAS
#define P_BIAS 1            // gyro bias in the filter
#endif
#ifndef P_ATT
#define P_ATT 1             // attitude error in the filter (needs P_FRAME_IMU)
#endif
#ifndef P_GYRO_NOISE
#define P_GYRO_NOISE 0.008
#endif
#ifndef P_PRED_DEPTH
#define P_PRED_DEPTH 3.0    // m: translational flow prediction depth (0 = rotation-only prediction)
#endif
#ifndef P_RHO0
#define P_RHO0 0.33        // prior inverse depth (1/m) of the flow-magnitude speed cue
#endif
#ifndef P_RHO_SIG
#define P_RHO_SIG 0         // its std (0 = cue off)
#endif
#ifndef P_RHO_NUPD
#define P_RHO_NUPD 12      // states the flow-magnitude cue corrects (3 = velocity only, 12 = all)
#endif
#ifndef P_SHRINK
#define P_SHRINK 0          // 1/s: output position decays toward the origin at this rate (0 = off)
#endif

#ifndef P_DET_REFINE
#define P_DET_REFINE 2      // hill-climb steps of the sparse corner search
#endif
#ifndef P_DET_REFINE_STEP
#define P_DET_REFINE_STEP 1
#endif
#ifndef P_TRACK_EVERY
#define P_TRACK_EVERY 1     // a feature is tracked every this many frames (staggered by id)
#endif
#ifndef P_MAX_MEAS
#define P_MAX_MEAS 100      // coplanarity measurements per frame
#endif
#ifndef P_HISTCORR
#define P_HISTCORR 1        // attitude corrections also rotate the stored history (bearings, IMU integrals)
#endif
#ifndef P_RHO_EVERY
#define P_RHO_EVERY 1       // a feature gives a flow-magnitude row every this many frames
#endif
#ifndef P_RHO_MEDIAN
#define P_RHO_MEDIAN 0      // >0: one flow-magnitude row per frame from the median feature (needs this many)
#endif
