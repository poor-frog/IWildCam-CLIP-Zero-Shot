# STMP-Adapter Kaggle Result

Date: 2026-07-08

## Method

**STMP-Adapter** is an eval-only extension of the current FLYP + WiSE +
validation-time Tail Prototype Adapter baseline. It uses iWildCam metadata to
apply sequence-level logit consensus after the prototype residual logits.

The run does not use DRM concept descriptions.

## Raw Artifact

```text
outputs_log/stmp_adapter_kaggle_2026-07-08.log
outputs_log/stmp_key_ablation_kaggle_2026-07-08.log
```

## Checkpoint

```text
/kaggle/input/datasets/thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint/flyp_nodrm_wise_vitb16_iwildcamval_best.pt
```

## Metadata Audit

Direct observation from the Kaggle log:

```text
metadata_fields = ['location', 'sequence', 'year', 'month', 'day', 'hour', 'minute', 'second', 'y', 'from_source_domain']
resolved_sequence_index = 1
resolved_camera_index = 0
resolved_datetime_index = None
```

This verifies that the sequence-consensus component used a real WILDS metadata
field named `sequence`.

## Validation Selection

Selected on `IWildCamVal` using `F1-macro_all`.

| Rank | Head | Scale | K | Seq eta | Tau | Tail gamma | Gate | Strength | Val Top-1 | Val F1 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 1 | prototype | 50 | 1 | 0.5 | 0 | 0 | margin | 0.25 | 61.09 | 41.93 |
| 2 | prototype_tau | 50 | 1 | 0.5 | 0 | 0 | margin | 0.25 | 61.09 | 41.93 |
| 3 | prototype | 50 | 1 | 0.5 | 0 | 0 | entropy | 1.0 | 61.15 | 41.92 |
| 5 | prototype | 50 | 8 | 0.5 | 0 | 0 | entropy | 1.0 | 58.56 | 41.86 |
| 11 | prototype | 50 | 1 | 0.5 | 0 | 0 | none | 0.0 | 61.31 | 41.58 |

`prototype_tau` is equivalent to `prototype` in this run because `tau=0`.

## Final Metrics

| Split | Default Top-1 | Default F1 | STMP Top-1 | STMP F1 | F1 Gain |
| --- | ---: | ---: | ---: | ---: | ---: |
| IWildCamIDVal | 76.55 | 44.89 | 79.59 | 50.53 | +5.64 |
| IWildCamVal | 54.31 | 36.94 | 61.09 | 41.93 | +4.99 |
| IWildCamID | 70.15 | 47.65 | 74.58 | 54.41 | +6.76 |
| IWildCamOOD | 70.12 | 31.74 | 75.73 | 38.01 | +6.27 |

## Key Final Ablation

The final key-ablation run evaluates the canonical candidates on all final
splits after validation selection. It confirms that most of the gain comes from
sequence consensus, not confidence gating or multi-prototype scoring.

| Candidate | K | Seq eta | Gate | Strength | IWildCamVal F1 | IWildCamIDVal F1 | IWildCamID F1 | IWildCamOOD F1 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| TPA baseline | 1 | 0 | none | 0 | 39.60 | 48.09 | 51.41 | 36.11 |
| STMP sequence-only | 1 | 0.5 | none | 0 | 41.58 | 50.37 | 54.61 | 37.98 |
| STMP selected-gate | 1 | 0.5 | margin | 0.25 | 41.93 | 50.53 | 54.41 | 38.01 |
| Multi-prototype sanity | 8 | 0.5 | entropy | 1.0 | 41.86 | 47.40 | 49.04 | 36.07 |

OOD contribution:

```text
TPA baseline        36.11
Sequence-only STMP  37.98  (+1.87 over TPA)
Selected-gate STMP  38.01  (+0.03 over sequence-only)
K=8 multi-prototype 36.07  (-0.04 vs TPA)
```

## Comparison to Previous Best TPA

| Method | IWildCamOOD F1 |
| --- | ---: |
| FLYP + Tail Prototype Adapter, first local baseline | 35.89 |
| FLYP + Tail Prototype Adapter, best observed Kaggle rerun | 36.11 |
| STMP-Adapter, this Kaggle run | 38.01 |

The predefined STMP acceptance threshold was:

```text
IWildCamOOD F1 > 36.11
```

The strong promotion threshold was:

```text
IWildCamOOD F1 >= 37.0
```

This run satisfies both thresholds.

## Decision

Promote **STMP-Adapter sequence-only** as the current main ViT-B/16 method
claim. Use selected-gate as a minor validation-selected ablation, not as the
core method.

The cleanest current claim is:

```text
Sequence-aware prototype adaptation improves FLYP + WiSE on iWildCam by using
camera-trap sequence metadata during validation-time adaptation.
```

Canonical method config for the main claim:

```text
prototype_scale = 50
K = 1
sequence_eta = 0.5
gate = none
tau = 0
tail_gamma = 0
selection_split = IWildCamVal
```

Selected-gate config:

```text
prototype_scale = 50
K = 1
sequence_eta = 0.5
gate = margin
gate_strength = 0.25
tau = 0
tail_gamma = 0
selection_split = IWildCamVal
```

Do not present confidence gating as the primary contribution. It improves OOD
F1 only from `37.98` to `38.01`. Do not present multi-prototype as a positive
contribution in the current ViT-B/16 setting. `K=8` drops to OOD F1 `36.07`.
