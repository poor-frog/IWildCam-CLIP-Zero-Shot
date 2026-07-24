# Paper-Grade Training Foundation v0

This experiment establishes a reproducible multi-seed training foundation before any new method development. The authoritative frozen protocol is [`preregistration.json`](preregistration.json).

The trained method is **PoorFrogs label-template FLYP**: it uses deterministic class-label prompt templates and a multi-positive CLIP contrastive loss. It is not an official FLYP reproduction because it does not use paired image-text training data. The repository's `--drm-weight` option is an L2-to-initialization surrogate and is locked to zero; it must not be described as official DRM.

Official DRM remains a separate single-checkpoint eval-only reference. STP remains a frozen post-hoc evaluation head and this experiment does not reopen sequence-method development.

## Firewall

- The three-seed pilot may access only IWildCam train and `IWildCamVal`.
- The two completion seeds use the same frozen configuration.
- `IWildCamIDVal`, `IWildCamID`, and `IWildCamOOD` remain unopened for these checkpoints until all five validation receipts and head configurations are frozen.
- CCT-20 remains unopened.
- No seed may be dropped or repeated because of its metric.

## Task ledger

| ID | Task | Status | Artifact |
| --- | --- | --- | --- |
| PGF-00 | Freeze claim boundary, configuration, seeds, firewall, statistics, and validity gate | done | `experiments/paper_grade_training_foundation_v0/preregistration.json` |
| PGF-01 | Implement deterministic seed controls and record the runtime determinism envelope | done | `src/training_determinism.py` |
| PGF-02 | Implement immutable per-seed receipts, provenance binding, and split-firewall enforcement | done | `src/models/paper_grade_training_foundation.py` |
| PGF-03 | Add a private three-seed pilot Kaggle launcher with unique checkpoint paths | done | `kaggle-paper-grade-training-foundation-v0/` |
| PGF-04 | Run local synthetic and package smoke tests without opening final splits | done | `tests/test_paper_grade_training_foundation_v0.py` |
| PGF-05 | Execute and verify the three-seed Val-only pilot | done | `outputs_log/kaggle-paper-grade-training-foundation-v0-pilot/` |
| PGF-05a | Partition the pilot into one immutable Kaggle output per seed | done | `kaggle-paper-grade-training-foundation-v0-seed-*/` |
| PGF-05b | Replay three seed artifacts and freeze the pilot manifest | done | `experiments/paper_grade_training_foundation_v0/pilot_manifest.json` |
| PGF-06 | Execute the two frozen completion seeds and freeze the validation manifest | in_progress | `experiments/paper_grade_training_foundation_v0/five_seed_validation_manifest.json` |
| PGF-07 | Run the one-shot five-seed final evaluation after PGF-06 | blocked | `experiments/paper_grade_training_foundation_v0/five_seed_final_evaluation.json` |
| PGF-08 | Apply the validity gate and write the final assessment | blocked | `experiments/paper_grade_training_foundation_v0/closure_receipt.json` |

PGF-06 is blocked on a valid PGF-05 pilot. PGF-07 and PGF-08 are blocked until a valid five-seed validation manifest exists.

The first PGF-05 attempt stopped before completing epoch 1 because the default AMP loss scale produced a non-finite gradient at seed `20260721`, batch 0. The metric-free implementation repair is frozen in [`implementation_repair_1.md`](implementation_repair_1.md). The repaired combined run then hit Kaggle's runtime boundary during the second seed and exported no artifacts; its scheduling repair is frozen in [`infrastructure_repair_2.md`](infrastructure_repair_2.md). All three independent pilot outputs subsequently passed their original gates and were sealed in `pilot_manifest.json`. The stage-2 completion dispatch guard is documented in [`infrastructure_repair_3.md`](infrastructure_repair_3.md).
