#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_full.py \
  --model=ViT-B/32 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
  --data-location=./data \
  --batch-size=32 \
  --workers=4 \
  --n-ctx=2 \
  --maple-prompt-depth=9 \
  --epochs=9 \
  --lr=0.002 \
  --wd=1e-5 \
  --val-dataset=IWildCamIDVal \
  --best-metric=F1-macro_all \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=maple-vit-b32 \
  --save=./checkpoints/maple_full_prompt_learner.pt
