# FLYP + Tail Prototype Adapter Reproduction

Date: 2026-07-05

## Method

**FLYP + Tail Prototype Adapter** is a validation-time adapter for the existing
FLYP + WiSE checkpoint. It adds class-prototype residual logits built from
training-set image features.

This run does not use DRM concept descriptions.

## Checkpoint

```text
checkpoints/flyp_nodrm_wise_vitb16_iwildcamval_best.pt
```

## Canonical Config

```text
model = ViT-B-16
selection_split = IWildCamVal
selection_metric = F1-macro_all
prototype_scale = 50
tau = 0
concept = disabled
train_cache = all train examples
```

## Command

```bash
./scripts/eval_tail_prototype_adapter_iwildcam.sh
```

Equivalent explicit command:

```bash
PYTHONPATH=. .venv/bin/python src/eval_tail_cache.py \
  --model=ViT-B-16 \
  --train-dataset=IWildCam \
  --val-dataset=IWildCamVal \
  --eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD \
  --template=iwildcam_template \
  --data-location=./data \
  --load=checkpoints/flyp_nodrm_wise_vitb16_iwildcamval_best.pt \
  --prototype-scale-grid=50 \
  --cache-tau-grid=0 \
  --max-cache-examples-per-class=0 \
  --batch-size=256 \
  --workers=8 \
  --device=auto \
  --wandb \
  --wandb-project=PoorFrogs \
  --wandb-run-name=flyp-nodrm-wise-tail-prototype-adapter-scale50-tau0
```

W&B run:

```text
https://wandb.ai/poorfrogs/PoorFrogs/runs/5yl9r0vh
```

## Validation Selection

Selected on `IWildCamVal`.

| Rank | Head | Scale | Tau | F1-macro | Top-1 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | prototype | 50 | 0 | 39.71 | 57.20 |
| 2 | prototype_tau | 50 | 0 | 39.71 | 57.20 |
| 3 | default | 0 | 0 | 36.94 | 54.31 |

`prototype_tau` is equivalent to `prototype` in this run because `tau=0`.

## Final Metrics

| Split | Head | Top-1 | F1-macro |
| --- | --- | ---: | ---: |
| IWildCamIDVal | default | 76.55 | 44.89 |
| IWildCamIDVal | prototype | 76.36 | 49.33 |
| IWildCamVal | default | 54.31 | 36.94 |
| IWildCamVal | prototype | 57.20 | 39.71 |
| IWildCamID | default | 70.15 | 47.65 |
| IWildCamID | prototype | 70.74 | 51.31 |
| IWildCamOOD | default | 70.12 | 31.74 |
| IWildCamOOD | prototype | 72.38 | 35.99 |

## Result Summary

| Split | F1 Gain | Top-1 Gain |
| --- | ---: | ---: |
| IWildCamIDVal | +4.44 | -0.19 |
| IWildCamVal | +2.77 | +2.89 |
| IWildCamID | +3.66 | +0.59 |
| IWildCamOOD | +4.25 | +2.26 |

## Decision

Promote this as the current ViT-B/16 baseline:

```text
FLYP + Tail Prototype Adapter
```

The run satisfies the predefined pass condition:

```text
IWildCamOOD F1 >= 35.5
```

Observed:

```text
IWildCamOOD F1 = 35.99
```

## Next Gate

Run the same script on Kaggle with the same checkpoint attached as an input
artifact. The Kaggle result should be considered reproducible if OOD F1 remains
at or above `35.5`.
