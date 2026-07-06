# Confidence-Gated Prototype Adapter Extension

Date: 2026-07-06

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
