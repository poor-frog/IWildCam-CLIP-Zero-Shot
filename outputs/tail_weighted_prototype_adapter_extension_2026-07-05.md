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
