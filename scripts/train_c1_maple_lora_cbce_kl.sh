#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_LOCATION="${DATA_LOCATION:-./data}"
KL_WEIGHT="${KL_WEIGHT:-0.1}"
KL_TEMPERATURE="${KL_TEMPERATURE:-1.0}"
MAPLE_LORA_GAMMA="${MAPLE_LORA_GAMMA:-1.0}"
CLASS_BIAS_SCALE_GRID="${CLASS_BIAS_SCALE_GRID:--2,-1,-0.5,0,0.5,1,2}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/c1_maple_lora_kl_vitb16_bs256.pt}"

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_maple_lora.py \
  --model=ViT-B/16 \
  --train-dataset=IWildCam \
  --eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD \
  --data-location="$DATA_LOCATION" \
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
  --class-bias-calibration \
  --class-bias-scale-grid="$CLASS_BIAS_SCALE_GRID" \
  --kl-weight="$KL_WEIGHT" \
  --kl-temperature="$KL_TEMPERATURE" \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=c1-maple-lora-kl-vit-b16-bs256 \
  --save="$SAVE_PATH"
