#!/usr/bin/env bash
# Build the stage 3 vio_ekf candidates (work/ekf/<name>.wasm). Each line is a full configuration on top of native/vio_ekf/params.h.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON="P_SPLIT=2 P_TRI_ITERS=4 P_CHI2_FAST=2 P_DETECT_STEP=3 P_DETECT_CELLS=1 P_MARGIN=3 P_LK_ITERS=4 P_CLONES=3 P_FLOATCOV=1"
bash scripts/build_ekf_variant.sh ekf_c3    $COMMON P_MAX_FEAT=20 P_DETECT_BELOW=16 &
bash scripts/build_ekf_variant.sh ekf_slam8 $COMMON P_MAX_FEAT=20 P_DETECT_BELOW=16 P_INIT_TILT=0.1 P_SLAM=8 P_SLAM_REL=0.3 P_RHO_SIGMA=0.3 P_VPRIOR_SIGMA=10 P_SLAM_MAXUPD=4 P_MAX_CAND=8 P_SLAM_INIT_MAX=1 &
bash scripts/build_ekf_variant.sh ekf_lite  $COMMON P_MAX_FEAT=12 P_DETECT_BELOW=10 P_MAX_UPDATE_FEAT=3 P_HALF=2 P_MIN_SCORE=10 &
bash scripts/build_ekf_variant.sh ekf_ultra P_SPLIT=2 P_TRI_ITERS=3 P_CHI2_FAST=4 P_DETECT_STEP=3 P_DETECT_CELLS=1 P_MARGIN=2 P_LK_ITERS=3 P_FLOATCOV=1 P_HALF=2 P_CLONES=3 P_MAX_FEAT=8 P_DETECT_BELOW=6 P_MAX_UPDATE_FEAT=2 &
bash scripts/build_ekf_variant.sh ekf_slam8b $COMMON P_MAX_FEAT=20 P_DETECT_BELOW=16 P_INIT_TILT=0.1 P_SLAM=8 P_SLAM_REL=0.3 P_RHO_SIGMA=0.2 P_VPRIOR_SIGMA=7 P_SLAM_MAXUPD=4 P_MAX_CAND=8 P_SLAM_INIT_MAX=1 &
wait
