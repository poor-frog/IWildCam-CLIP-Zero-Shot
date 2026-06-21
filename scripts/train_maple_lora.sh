#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

MAPLE_LORA_GAMMA="${MAPLE_LORA_GAMMA:-1.0}"

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_lora.py \
  --model=ViT-B/16 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
  --data-location=./data \
  --batch-size=256 \
  --workers=4 \
  --n-ctx=2 \
  --maple-prompt-depth=9 \
  --maple-lora-rank=4 \
  --maple-lora-alpha=8 \
  --maple-lora-layers=last6 \
  --maple-lora-gamma="$MAPLE_LORA_GAMMA" \
  --epochs=20 \
  --lr=1e-5 \
  --wd=0.2 \
  --val-dataset=IWildCamVal \
  --best-metric=F1-macro_all \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=maple-lora-vit-b16-bs256-r4-last6-e20-lr1e-5 \
  --save=./checkpoints/maple_lora_vitb16_bs256_r4_last6_e20_lr1e-5.pt
