#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_full.py \
  --model=ViT-B/16 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
  --data-location=./data \
  --batch-size=256 \
  --workers=4 \
  --n-ctx=2 \
  --maple-prompt-depth=9 \
  --epochs=20 \
  --lr=1e-5 \
  --wd=0.2 \
  --val-dataset=IWildCamVal \
  --best-metric=F1-macro_all \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=maple-full-vit-b16-bs256-iwildcamval \
  --save=./checkpoints/maple_full_prompt_learner_vitb16_bs256_iwildcamval.pt
