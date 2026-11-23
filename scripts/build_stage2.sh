#!/usr/bin/env bash
# Stage 2 (Nov 22): the metric changed from Sim(3) ATE to unaligned ATE, so the step length must be in metres.
# Builds the C++ variants measured that day; each is the native/klt_bounded code plus parameter overrides.
#   bash scripts/build_stage2.sh            # submission candidates r4m and r5_p0.42
#   bash scripts/build_stage2.sh sweep      # also every variant in results/klt-bounded/nov22_stage2.json
# Evaluate with: bash scripts/eval_cpp.sh 10 5 <names...>
set -euo pipefail
cd "$(dirname "$0")/.."
# n4 (main line) with the gyro-predicted LK start (g4) and the learned IMU speed model (r4)
R4="P_BIAS_BASELINE=30 P_FAST_GRAY=-1 P_MIN_TRACKS=12 P_MIN_INLIERS=8 P_MAX_CORNERS=20 P_E_MAX_POINTS=16 P_KR_HYPOTHESES=8
    P_REPLENISH_FRACTION=0.85 P_DETECT_STRIPS=3 P_LK_PREDICT=1 P_LK_WIN=5 P_LK_ITERS=5 P_SPEED_MODEL=1"
# k7 (low-fuel line, 1/8 resolution) with the learned IMU speed model (r5)
R5="P_SCALE=0.125 P_FAST_GRAY=-1 P_GRAY_ROWS_8=2 P_BIAS_BASELINE=30 P_MIN_DISTANCE=2.0 P_LK_WIN=5 P_DETECT_STRIPS=3
    P_MAX_CORNERS=16 P_E_MAX_POINTS=16 P_KR_HYPOTHESES=8 P_MIN_TRACKS=8 P_MIN_INLIERS=6 P_REPLENISH_FRACTION=0.7 P_SPEED_MODEL=1"
# P_SPEED_PRIOR is the mean speed the learned model is rescaled to. 0.3246063005 is the training mean of the public
# episodes (SPEED_MEAN_PUBLIC), i.e. no rescaling: speed in m/s. The 1/8 build needs more, because its noisier
# directions shorten the net path; 0.42 was best on the public set (swept 0.3246-0.55).
bash scripts/build_cpp_variant.sh r4m $R4 P_SPEED_PRIOR=0.3246063005
bash scripts/build_cpp_variant.sh r5_p0.42 $R5 P_SPEED_PRIOR=0.42
if [ "${1:-}" = sweep ]; then
  bash scripts/build_cpp_variant.sh r4 $R4
  for p in 0.37 0.42; do bash scripts/build_cpp_variant.sh r4_p$p $R4 P_SPEED_PRIOR=$p; done
  bash scripts/build_cpp_variant.sh r5 $R5
  bash scripts/build_cpp_variant.sh r5m $R5 P_SPEED_PRIOR=0.3246063005
  for p in 0.37 0.40 0.47 0.55; do bash scripts/build_cpp_variant.sh r5_p$p $R5 P_SPEED_PRIOR=$p; done
  bash scripts/build_cpp_variant.sh n4m ${R4/P_SPEED_MODEL=1/P_SPEED_MODEL=0} P_SPEED_PRIOR=0.3246063005
fi
