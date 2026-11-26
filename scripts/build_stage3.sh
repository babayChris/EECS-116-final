#!/usr/bin/env bash
# Build the stage 3 vio_ekf candidates (work/ekf/<name>.wasm). Each line is a full configuration on top of
# native/vio_ekf/params.h. Priors tuned on work/data/s3 scenes 12 and 45 (scenes 88 and 150 held out).
set -euo pipefail
cd "$(dirname "$0")/.."
PRI="P_VPRIOR_SIGMA=3 P_RHO_SIGMA=0.2 P_INIT_BG=0.01 P_GYRO_WALK=1e-4"
MAIN="P_SPLIT=2 P_TRI_ITERS=4 P_CHI2_FAST=2 P_DETECT_STEP=3 P_DETECT_CELLS=1 P_MARGIN=3 P_LK_ITERS=4 P_CLONES=3 P_FLOATCOV=1"
bash scripts/build_ekf_variant.sh ekf_main  $MAIN $PRI P_MAX_FEAT=20 P_DETECT_BELOW=16 &
bash scripts/build_ekf_variant.sh ekf_lite  $MAIN $PRI P_MAX_FEAT=12 P_DETECT_BELOW=10 P_MAX_UPDATE_FEAT=3 P_HALF=2 &
bash scripts/build_ekf_variant.sh ekf_ultra P_SPLIT=2 P_TRI_ITERS=3 P_CHI2_FAST=4 P_DETECT_STEP=3 P_DETECT_CELLS=1 P_MARGIN=2 P_LK_ITERS=3 P_FLOATCOV=1 P_HALF=2 P_CLONES=3 P_MAX_FEAT=8 P_DETECT_BELOW=6 P_MAX_UPDATE_FEAT=2 $PRI &
# low-fuel EKF (Nov 26 afternoon): 2 clones, sparse corner search, structured propagation (P_FAST), covariance only at
# clone frames (P_COV_DEFER), mean once per frame. ekf_sliced was submitted as "m_d" (4.76 m, 230k on the server).
C2="P_SPLIT=2 P_TRI_ITERS=3 P_CHI2_FAST=4 P_DETECT_STEP=3 P_DETECT_CELLS=1 P_MARGIN=2 P_LK_ITERS=3 P_FLOATCOV=1 P_HALF=2 P_MAX_FEAT=8 P_DETECT_BELOW=6 P_MAX_UPDATE_FEAT=2 P_PHI_FRAME=1 P_CLONES=2 P_MIN_OBS=2"
bash scripts/build_ekf_variant.sh ekf_c2     $C2 $PRI P_DET_CAND=16 &
bash scripts/build_ekf_variant.sh ekf_sliced $C2 $PRI P_FAST=2 P_DET_CAND=9 P_DET_REFINE=1 P_MAX_CAND=3 P_MEAN_FRAME=1 P_COV_DEFER=1 &
# before tuning on the stage-3-like set (for comparison)
bash scripts/build_ekf_variant.sh ekf_c3    $MAIN P_MAX_FEAT=20 P_DETECT_BELOW=16 &
wait
