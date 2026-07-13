# PoorFrogs Results

This file records PoorFrogs-local IWildCam result provenance and reproduced metrics.

`/Users/jky/Downloads/FLYP/FLYP` is history/reference only. Numbers from that repo must not be treated as PoorFrogs-reproduced results unless a PoorFrogs evidence file or rerun explicitly reproduces them.

For corrected OOD-generalization studies, use `IWildCamVal` (WILDS `val`) for checkpoint, tau, lambda, learning-rate, epoch, model, and hyperparameter selection. `IWildCamOOD` is final-test-only and must not be used for any selection or tuning decision.

## Current best observed ViT-B/16 result

The current best observed ViT-B/16 result is **DRM + WiSE + STP** on a
converted official DRM checkpoint. WiSE and STP are selected on `IWildCamVal`;
`IWildCamOOD` is final-test-only.

Evidence file:

```text
outputs/drm_wise_stp_kaggle_2026-07-12.md
```

| Method | IWildCamVal F1 | IWildCamIDVal F1 | IWildCamID F1 | IWildCamOOD F1 |
| ------ | -------------- | ---------------- | ------------- | -------------- |
| Official DRM + WiSE, dual concept inference | n/a | 51.36% | 54.82% | 38.91% |
| DRM + WiSE, converted no TPA | 39.64% | 46.73% | 53.10% | 38.77% |
| DRM + WiSE + TPA, converted | 41.39% | 51.69% | 56.09% | 40.46% |
| DRM + WiSE + STP, converted | **43.70%** | **52.98%** | **57.95%** | **42.42%** |
| FLYP + WiSE default logits | 36.94% | 44.89% | 47.65% | 31.74% |
| FLYP + Tail Prototype Adapter | 39.60% | 48.09% | 51.41% | 36.11% |
| STMP-Adapter sequence-only | 41.58% | 50.37% | 54.61% | 37.98% |
| STMP-Adapter selected-gate ablation | 41.93% | 50.53% | 54.41% | 38.01% |
| STMP-Adapter multi-prototype sanity | 41.86% | 47.40% | 49.04% | 36.07% |

The previous FLYP-only STMP evidence remains documented in
`outputs/stmp_adapter_kaggle_2026-07-08.md`. The current DRM comparison below
is a separate matched control study; the official DRM row must not be treated
as the same protocol as the converted-DRM rows.

### Matched converted-DRM TPA control

The matched no-TPA and TPA runs use the same converted checkpoint, the same
`IWildCamVal` selection split, the same WiSE alpha grid (`0.0` through `0.9`),
and no concept descriptions. The no-TPA control sets the prototype residual
scale to `0`; TPA uses `prototype_scale=50`, `tau=0`, `K=1`, and
`sequence_eta=0`.

| Method | Selected alpha | IWildCamVal F1 | IWildCamIDVal F1 | IWildCamID F1 | IWildCamOOD F1 |
| ------ | -------------- | -------------- | ---------------- | ------------- | -------------- |
| Official DRM + WiSE, dual concept inference | 0.3 | n/a | 51.36% | 54.82% | 38.91% |
| Converted DRM + WiSE, no TPA | 0.1 | 39.64% | 46.73% | 53.10% | 38.77% |
| Converted DRM + WiSE + TPA | 0.2 | **41.39%** | **51.69%** | **56.09%** | **40.46%** |
| Converted DRM + WiSE + STP | 0.2 | **43.70%** | **52.98%** | **57.95%** | **42.42%** |

W&B provenance:

- Official DRM + WiSE: `sq3so2ub`
- Matched no-TPA final: `val92el5`
- Matched TPA final: `kv7qntgy`
- DRM + WiSE + STP: `03gg65vx`

Within the matched converted-DRM protocol, TPA improves over no-TPA by
`+1.75` validation F1 and `+1.69` OOD F1. This validates TPA as a useful
control adapter, while STP remains the current best observed converted-DRM
method at `42.42%` OOD F1. The official DRM row uses its own official
`DRM_eval.py` dual-inference path and selects on `IWildCamIDVal`; it is kept
as a reference baseline rather than a direct matched comparison.

## BTEL v1 negative ablation

`FLYP + BTEL + WiSE` with `btel_weight=0.01`, negative quantile `0.95`, and
prototype scale `50` reached 33.52% `IWildCamOOD` macro-F1. It slightly
improved OOD over the logged FLYP + WiSE baseline but regressed both ID splits;
do not promote BTEL v1 as the main method. Full provenance is recorded in
`outputs/drm_wise_stp_kaggle_2026-07-12.md`.

## Historical/reference-only metrics

Any tau-sweep metric inherited from old notes or `/Users/jky/Downloads/FLYP/FLYP` is reference-only until reproduced in `/Users/jky/Downloads/FLYP/PoorFrogs`. Do not use reference-only numbers to claim PoorFrogs completion, checkpoint superiority, or OOD improvement.
