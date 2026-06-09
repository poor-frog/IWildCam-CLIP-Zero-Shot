#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_lora.py \
  --model=ViT-B/32 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
  --data-location=./data \
  --batch-size=32 \
  --workers=4 \
  --n-ctx=2 \
  --maple-prompt-depth=9 \
  --maple-lora-rank=8 \
  --maple-lora-alpha=16 \
  --maple-lora-layers=last6 \
  --epochs=9 \
  --lr=0.002 \
  --wd=1e-5 \
  --val-dataset=IWildCamIDVal \
  --best-metric=F1-macro_all \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=maple-lora-vit-b32-r8-last6 \
  --save=./checkpoints/maple_lora_r8_last6.pt
