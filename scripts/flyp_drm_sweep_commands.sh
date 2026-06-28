#!/usr/bin/env bash
set -euo pipefail

WISE_ALPHAS="${WISE_ALPHAS:-0.0,0.05,0.1,0.15,0.2}"
DRM_WEIGHTS="${DRM_WEIGHTS:-0.0 0.1 0.5 1.0}"

for drm_weight in $DRM_WEIGHTS; do
  run_suffix="drm${drm_weight//./p}"
  echo "python kaggle_main.py --mode=flyp --drm-weight=${drm_weight} --wise-alphas=${WISE_ALPHAS} --wandb-run-name=flyp-${run_suffix}-wise-fine-vit-b16-iwildcamval --save=/kaggle/working/checkpoints/flyp_${run_suffix}_wise_fine_vitb16_iwildcamval.pt"
done
