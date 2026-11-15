// Settings of the C++ klt_bounded port; defaults are the Python e5_p05_base3 variant (solutions/klt_bounded/variants.json).
// A variant is a copy of this directory with different values here.
#pragma once
#ifndef P_SCALE
#define P_SCALE 0.25
#endif
#ifndef P_MAX_CORNERS
#define P_MAX_CORNERS 40
#endif
#ifndef P_QUALITY
#define P_QUALITY 0.01
#endif
#ifndef P_MIN_DISTANCE
#define P_MIN_DISTANCE 3.0
#endif
#ifndef P_BLOCK_SIZE
#define P_BLOCK_SIZE 3
#endif
#ifndef P_LK_WIN
#define P_LK_WIN 7
#endif
#ifndef P_LK_LEVELS
#define P_LK_LEVELS 1
#endif
#ifndef P_LK_ITERS
#define P_LK_ITERS 10
#endif
#ifndef P_LK_EPS
#define P_LK_EPS 0.03
#endif
#ifndef P_LK_FB_PX
#define P_LK_FB_PX 0.0  // 0: no forward-backward check
#endif
#ifndef P_E_MAX_POINTS
#define P_E_MAX_POINTS 30
#endif
#ifndef P_MIN_TRACKS
#define P_MIN_TRACKS 20
#endif
#ifndef P_MIN_INLIERS
#define P_MIN_INLIERS 10
#endif
#ifndef P_KR_HYPOTHESES
#define P_KR_HYPOTHESES 32
#endif
#ifndef P_KR_THRESHOLD_PX
#define P_KR_THRESHOLD_PX 1.0
#endif
#ifndef P_MIN_DEROTATED_PARALLAX
#define P_MIN_DEROTATED_PARALLAX 0.25
#endif
#ifndef P_SPEED_PRIOR
#define P_SPEED_PRIOR 0.5
#endif
#ifndef P_REPLENISH_FRACTION
#define P_REPLENISH_FRACTION 0.5
#endif
#ifndef P_BASELINE_FRAMES
#define P_BASELINE_FRAMES 3
#endif
#ifndef P_WARMUP
#define P_WARMUP 1
#endif
#ifndef P_FAST_GRAY
#define P_FAST_GRAY 0  // 0: OpenCV cvtColor + INTER_AREA resize; k > 0: direct block luma, every k-th pixel of each block
#endif
#ifndef P_BIAS_BASELINE
#define P_BIAS_BASELINE 0  // 0: no gyro bias estimation from vision; else frames between the compared views
#endif
#ifndef P_BIAS_GAIN
#define P_BIAS_GAIN 0.02
#endif
#ifndef P_BIAS_RIDGE
#define P_BIAS_RIDGE 1e-6
#endif
#ifndef P_BIAS_MAX_STEP_DPS
#define P_BIAS_MAX_STEP_DPS 0.2
#endif
