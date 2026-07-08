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

Promote **STMP-Adapter** as the current main ViT-B/16 method.

The cleanest current claim is:

```text
Sequence-aware prototype adaptation improves FLYP + WiSE on iWildCam by using
camera-trap sequence metadata during validation-time adaptation.
```

Do not present confidence gating as the primary contribution yet. The selected
gate improves validation F1 only slightly over the no-gate sequence candidate:

```text
gate=margin,strength=0.25: Val F1 = 41.93
gate=none:                 Val F1 = 41.58
```

Do not present multi-prototype as the primary contribution yet. The selected
configuration used `K=1`; `K=8` was competitive on validation but did not win.

## Next Ablation

Run a focused STMP ablation to isolate the contribution:

| Ablation | Expected purpose |
| --- | --- |
| `eta=0, K=1, gate=none` | Original TPA baseline under the same launcher |
| `eta=0.5, K=1, gate=none` | Sequence-only STMP |
| `eta=0.5, K=1, gate=margin,strength=0.25` | Selected STMP |
| `eta=0.5, K=8, gate=entropy,strength=1.0` | Multi-prototype sanity check |

If sequence-only is close to selected STMP on OOD, the method framing should
center on sequence consensus rather than gate tuning.
