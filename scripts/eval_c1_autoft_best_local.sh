#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATA_LOCATION="${DATA_LOCATION:-./data}"
CKPT="${CKPT:-./checkpoints/c1_autoft_1k_oodval_vitb16_bs256_best.pt}"
EVAL_DATASETS="${EVAL_DATASETS:-IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD}"
SELECTION_SPLIT="${SELECTION_SPLIT:-IWildCamVal}"
TAU_GRID="${TAU_GRID:-}"
CLASS_BIAS_SCALE_GRID="${CLASS_BIAS_SCALE_GRID:--2,-1,-0.5,0,0.5,1,2}"
MAPLE_LORA_GAMMA="${MAPLE_LORA_GAMMA:-1.0}"
NUM_OOD_HP_EXAMPLES="${NUM_OOD_HP_EXAMPLES:-1000}"
CLASS_BALANCED_OOD="${CLASS_BALANCED_OOD:-1}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-}"
DEVICE="${DEVICE:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "./.venv/bin/python" ]]; then
    PYTHON_BIN="./.venv/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi

args=(
  --model=ViT-B/16
  --train-dataset=IWildCam
  --template=iwildcam_template
  --eval-datasets="$EVAL_DATASETS"
  --data-location="$DATA_LOCATION"
  --device="$DEVICE"
  --batch-size="$BATCH_SIZE"
  --workers="$WORKERS"
  --n-ctx=2
  --ctx-init="a photo of a"
  --maple-prompt-depth=9
  --maple-lora-rank=4
  --maple-lora-alpha=8
  --maple-lora-layers=last6
  --maple-lora-gamma="$MAPLE_LORA_GAMMA"
  --epochs=0
  --load="$CKPT"
  --selection-split="$SELECTION_SPLIT"
  --num-ood-hp-examples="$NUM_OOD_HP_EXAMPLES"
  --class-bias-calibration
  --class-bias-scale-grid="$CLASS_BIAS_SCALE_GRID"
  --best-metric=F1-macro_all
)

if [[ -n "$TAU_GRID" ]]; then
  args+=(--logit-adjustment-tau-grid="$TAU_GRID")
fi

if [[ "$CLASS_BALANCED_OOD" == "1" || "$CLASS_BALANCED_OOD" == "true" ]]; then
  args+=(--class-balanced-ood)
else
  args+=(--no-class-balanced-ood)
fi

if [[ -n "$MAX_EVAL_BATCHES" ]]; then
  args+=(--max-eval-batches="$MAX_EVAL_BATCHES")
fi

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
  printf 'KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. %q src/train_maple_full.py' "$PYTHON_BIN"
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. "$PYTHON_BIN" src/train_maple_full.py "${args[@]}"
