#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_coop.py \
  --model=ViT-B/32 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
  --data-location=./data \
  --batch-size=32 \
  --workers=4 \
  --n-ctx=16 \
  --ctx-init="a photo of a" \
  --class-token-position=end \
  --epochs=50 \
  --lr=0.002 \
  --wd=1e-5 \
  --save=./checkpoints/coop_prompt_learner.pt
