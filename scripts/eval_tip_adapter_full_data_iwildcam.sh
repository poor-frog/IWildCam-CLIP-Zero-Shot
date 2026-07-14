#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the converted DRM CLIPEncoder checkpoint.}"
DATA_LOCATION="${DATA_LOCATION:-./data}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-2}"

PYTHONPATH=. .venv/bin/python src/eval_tail_cache.py \
  --model=ViT-B-16 \
  --train-dataset=IWildCam \
  --val-dataset=IWildCamVal \
  --eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD \
  --template=iwildcam_template \
  --data-location="${DATA_LOCATION}" \
  --load="${CHECKPOINT}" \
  --wise-eval-alpha=0.2 \
  --prototype-scale-grid=0 \
  --cache-tau-grid=0 \
  --tail-gamma-grid=0 \
  --gate-mode-grid=none \
  --gate-strength-grid=0 \
  --sequence-consensus-grid=0 \
  --sctr-strength-grid=0 \
  --sctr-tail-protection-grid=0 \
  --multi-prototype-k-grid=1 \
  --max-cache-examples-per-class=0 \
  --tip-adapter-beta-grid=0.1,0.5,1,2,5,7 \
  --tip-adapter-alpha-grid=0.1,0.5,1,2,3 \
  --tip-adapter-query-chunk-size="${BATCH_SIZE}" \
  --tip-adapter-cache-chunk-size=16384 \
  --summary-head=tip_adapter \
  --batch-size="${BATCH_SIZE}" \
  --workers="${WORKERS}" \
  --device="${DEVICE}" \
  --no-wandb
