#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:-checkpoints/flyp_nodrm_wise_vitb16_iwildcamval_best.pt}"
DATA_LOCATION="${DATA_LOCATION:-./data}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-8}"
WANDB_FLAG="${WANDB_FLAG:---wandb --wandb-project=PoorFrogs --wandb-run-name=flyp-nodrm-wise-tail-prototype-adapter-scale50-tau0}"


PYTHONPATH=. .venv/bin/python src/eval_tail_cache.py \
  --model=ViT-B-16 \
  --train-dataset=IWildCam \
  --val-dataset=IWildCamVal \
  --eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD \
  --template=iwildcam_template \
  --data-location="${DATA_LOCATION}" \
  --load="${CHECKPOINT}" \
  --prototype-scale-grid=50 \
  --cache-tau-grid=0 \
  --max-cache-examples-per-class=0 \
  --batch-size="${BATCH_SIZE}" \
  --workers="${WORKERS}" \
  --device="${DEVICE}" \
  ${WANDB_FLAG}
