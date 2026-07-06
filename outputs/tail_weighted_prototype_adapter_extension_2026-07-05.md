# Tail-Weighted Prototype Adapter Extension

Date: 2026-07-05

## Motivation

The confirmed baseline is **FLYP + Tail Prototype Adapter**:

```text
Kaggle IWildCamOOD F1 = 35.78
Local IWildCamOOD F1 = 35.99
```

The next method extension keeps the same validation-time setup and only changes
the prototype residual by class frequency.

## Method

Baseline residual:

```text
logits = zero_shot_logits + prototype_scale * prototype_logits
```

Tail-weighted residual:

```text
logits_c = zero_shot_logits_c + prototype_scale * w_c * prototype_logits_c
```

Class weights:

```text
w_c = normalize(clamp((count_c / max_count)^(-tail_gamma), max=tail_weight_max))
```

Default settings:

```text
prototype_scale = 50
tau = 0
tail_gamma in {0, 0.25, 0.5, 1.0}
tail_weight_max = 5.0
selection_split = IWildCamVal
selection_metric = F1-macro_all
concept = disabled
```

`tail_gamma=0` is exactly the confirmed TPA baseline.

## Run Command

```bash
./scripts/eval_tail_weighted_prototype_adapter_iwildcam.sh
```

Disable W&B:

```bash
WANDB_FLAG="--no-wandb" ./scripts/eval_tail_weighted_prototype_adapter_iwildcam.sh
```

## Kaggle Command Surface

The Kaggle kernel uses:

```text
kaggle_eval_tail_prototype_adapter.py
```

Default Kaggle extension grid:

```text
tail_gamma_grid = 0,0.25,0.5,1.0
tail_weight_max = 5.0
workers = 2
```

The kernel metadata should attach:

```text
thanhquang71/iwildcam-v2-0-2020-wilds-dataset
thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint
```

## Acceptance

Keep the extension only if it improves over the Kaggle-confirmed baseline:

```text
IWildCamOOD F1 > 35.78
```

Strong result:

```text
IWildCamOOD F1 >= 37.0
```

If no `tail_gamma > 0` candidate wins on `IWildCamVal`, keep the original
validation-time Tail Prototype Adapter as the main method.

## Kaggle Result

Kaggle kernel:

```text
thanhquang71/poorfrogs-tail-weighted-tpa-iwildcam
```

Validation selection on `IWildCamVal`:

| Rank | Head | Scale | Tau | Tail gamma | Top-1 | F1-macro |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | prototype | 50 | 0 | 0 | 57.22 | 39.60 |
| 2 | prototype_tau | 50 | 0 | 0 | 57.22 | 39.60 |
| 3 | default | 0 | 0 | 0 | 54.31 | 36.94 |
| 4 | prototype | 50 | 0 | 1.0 | 43.44 | 34.70 |
| 5 | prototype_tau | 50 | 0 | 1.0 | 43.44 | 34.70 |
| 6 | prototype | 50 | 0 | 0.5 | 36.74 | 32.98 |
| 7 | prototype_tau | 50 | 0 | 0.5 | 36.74 | 32.98 |
| 8 | prototype | 50 | 0 | 0.25 | 14.36 | 21.01 |
| 9 | prototype_tau | 50 | 0 | 0.25 | 14.36 | 21.01 |

Final metrics for the validation-selected head:

| Split | Default Top-1 | Default F1 | TPA Top-1 | TPA F1 |
| --- | ---: | ---: | ---: | ---: |
| IWildCamIDVal | 76.55 | 44.89 | 76.25 | 48.09 |
| IWildCamVal | 54.31 | 36.94 | 57.22 | 39.60 |
| IWildCamID | 70.15 | 47.65 | 70.70 | 51.41 |
| IWildCamOOD | 70.12 | 31.74 | 72.20 | 36.11 |

## Decision

Do not promote Tail-Weighted Prototype Adapter.

The best validation-selected setting is:

```text
tail_gamma = 0
```

That is exactly the original **FLYP + Tail Prototype Adapter** baseline. All
`tail_gamma > 0` settings underperform on the validation split, with
`tail_gamma=0.25` collapsing severely.

Record this as a negative ablation:

```text
Tail-weighted residual by inverse class frequency: failed.
Main method remains FLYP + Tail Prototype Adapter.
Best observed Kaggle IWildCamOOD F1 = 36.11.
```

The next confidence-gated validation-time extension was also neutral, so stop
adding inference-time adapter variants for now. Move to train-time
**Tail-Aware FLYP** with a frozen prototype auxiliary objective selected only on
`IWildCamVal`.
