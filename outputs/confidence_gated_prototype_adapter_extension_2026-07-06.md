# Confidence-Gated Prototype Adapter Extension

Date: 2026-07-06

Status: neutral ablation, not promoted.

## Motivation

The confirmed main method is **FLYP + Tail Prototype Adapter**:

```text
Best observed Kaggle IWildCamOOD F1 = 36.11
```

The previous tail-weighted extension failed because direct inverse-frequency
class weighting damaged validation calibration. This extension avoids class
frequency multipliers and instead gates the prototype residual per sample.

## Method

Baseline TPA:

```text
logits = zero_shot_logits + prototype_scale * prototype_logits
```

Confidence-gated TPA:

```text
logits = zero_shot_logits + alpha(x) * prototype_scale * prototype_logits
```

Gate:

```text
alpha(x) = clamp_min(1 + gate_strength * (uncertainty(x) - 0.5), 0)
```

This reduces the prototype residual for high-confidence samples and increases it
for uncertain samples while keeping `gate_strength=0` identical to the original
TPA baseline.

Uncertainty options:

```text
entropy: normalized entropy of zero-shot probabilities
margin:  1 - top2_probability_margin
```

## Config

```text
prototype_scale = 50
tau = 0
tail_gamma = 0
gate_mode in {none, entropy, margin}
gate_strength in {0, 0.25, 0.5, 1.0}
selection_split = IWildCamVal
selection_metric = F1-macro_all
concept = disabled
```

`gate_mode=none` or `gate_strength=0` is exactly the confirmed TPA baseline.

## Local Command

```bash
./scripts/eval_confidence_gated_prototype_adapter_iwildcam.sh
```

Disable W&B:

```bash
WANDB_FLAG="--no-wandb" ./scripts/eval_confidence_gated_prototype_adapter_iwildcam.sh
```

## Kaggle Command Surface

The Kaggle kernel uses:

```text
kaggle_eval_tail_prototype_adapter.py
```

Default Kaggle grid:

```text
gate_mode_grid = none,entropy,margin
gate_strength_grid = 0,0.25,0.5,1.0
tail_gamma_grid = 0
workers = 2
```

The kernel metadata should attach:

```text
thanhquang71/iwildcam-v2-0-2020-wilds-dataset
thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint
```

## Acceptance

Keep the extension only if it improves over the best observed Kaggle baseline:

```text
IWildCamOOD F1 > 36.11
```

Strong result:

```text
IWildCamOOD F1 >= 37.0
```

If no confidence-gated candidate beats `gate_mode=none` on `IWildCamVal`, keep
the original **FLYP + Tail Prototype Adapter** as the main method.

## Kaggle Result

Source log:

```text
outputs_log/Confidence_Gate_TPA_logs.txt
```

Kaggle command surface:

```text
kaggle_eval_tail_prototype_adapter.py
```

Important runtime note:

```text
The downloaded log shows --no-wandb, so this run did not sync to W&B.
```

Validation selection on `IWildCamVal`:

| Rank | Head | Scale | Tau | Tail gamma | Gate | Strength | Top-1 | F1-macro |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | prototype | 50 | 0 | 0 | margin | 1.0 | 57.20 | 39.79 |
| 2 | prototype_tau | 50 | 0 | 0 | margin | 1.0 | 57.20 | 39.79 |
| 3 | prototype | 50 | 0 | 0 | margin | 0.5 | 57.24 | 39.76 |
| 4 | prototype_tau | 50 | 0 | 0 | margin | 0.5 | 57.24 | 39.76 |
| 5 | prototype | 50 | 0 | 0 | entropy | 1.0 | 57.45 | 39.71 |
| 13 | prototype | 50 | 0 | 0 | none | 0.0 | 57.22 | 39.60 |

Final metrics for the validation-selected confidence-gated head:

| Split | Default Top-1 | Default F1 | Confidence-Gated TPA Top-1 | Confidence-Gated TPA F1 |
| --- | ---: | ---: | ---: | ---: |
| IWildCamIDVal | 76.55 | 44.89 | 76.37 | 48.28 |
| IWildCamVal | 54.31 | 36.94 | 57.20 | 39.79 |
| IWildCamID | 70.15 | 47.65 | 70.95 | 51.32 |
| IWildCamOOD | 70.12 | 31.74 | 72.10 | 35.88 |

## Decision

Do not promote Confidence-Gated TPA as the main method.

The selected gate is:

```text
gate_mode = margin
gate_strength = 1.0
```

It slightly improves validation F1 over the no-gate TPA candidate:

```text
IWildCamVal F1: 39.60 -> 39.79
```

However, the final OOD result is:

```text
IWildCamOOD F1 = 35.88
```

This does not beat the best observed Kaggle **FLYP + Tail Prototype Adapter**
result:

```text
Best observed Kaggle IWildCamOOD F1 = 36.11
```

Record this as a neutral ablation:

```text
Confidence-Gated TPA: neutral.
Best validation config: margin gate, strength=1.0.
Final IWildCamOOD F1 = 35.88.
Main method remains FLYP + Tail Prototype Adapter.
```

## Next Method Direction

Stop extending validation-time inference adapters for now. Tail weighting and
confidence gating did not produce a stronger OOD result than the confirmed TPA
baseline.

Move to **train-time Tail-Aware FLYP**:

```text
L = L_FLYP + lambda_tail * CE(prototype_scale * image_features @ class_prototypes.T, labels)
```

Recommended first grid:

```text
lambda_tail in {0.01, 0.03, 0.1}
prototype_scale in {20, 50, 100}
class_prototypes = frozen train-set image prototypes
concept descriptions = disabled
selection_split = IWildCamVal
final_split = IWildCamOOD
```

Keep the train-time extension only if it beats the validation-time adapter:

```text
IWildCamOOD F1 > 35.88
```

Strong promotion threshold:

```text
IWildCamOOD F1 >= 37.0 on ViT-B/16
```
