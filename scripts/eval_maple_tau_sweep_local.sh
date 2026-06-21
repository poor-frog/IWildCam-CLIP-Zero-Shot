#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_LOCATION="${DATA_LOCATION:-./data}"
CKPT="${CKPT:-./checkpoints/maple_full_prompt_learner_best.pt}"
TAU_GRID="${TAU_GRID:-0,0.25,0.5,0.75,1,1.5,2}"
CLASS_BIAS_SCALE_GRID="${CLASS_BIAS_SCALE_GRID:--2,-1,-0.5,0,0.5,1,2}"

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_full.py \
  --model=ViT-B/32 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD \
  --data-location="$DATA_LOCATION" \
  --batch-size=32 \
  --workers=4 \
  --n-ctx=2 \
  --maple-prompt-depth=9 \
  --epochs=0 \
  --load="$CKPT" \
  --selection-split=IWildCamVal \
  --logit-adjustment-tau-grid="$TAU_GRID" \
  --class-bias-calibration \
  --class-bias-scale-grid="$CLASS_BIAS_SCALE_GRID" \
  --best-metric=F1-macro_all
