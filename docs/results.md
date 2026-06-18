# PoorFrogs Results

This file records PoorFrogs-local IWildCam result provenance. It separates existing PoorFrogs checkpoint inventory, locally documented/reproduced metrics, historical reference-only metrics, and metrics that remain unknown or unverified.

`/Users/jky/Downloads/FLYP/FLYP` is history/reference only. Numbers from that repo must not be treated as PoorFrogs-reproduced results unless a PoorFrogs evidence file or rerun explicitly reproduces them.

For corrected OOD-generalization studies, use `IWildCamVal` (WILDS `val`) for checkpoint, tau, lambda, learning-rate, epoch, model, and hyperparameter selection. `IWildCamOOD` is final-test-only and must not be used for any selection or tuning decision.

## PoorFrogs checkpoint inventory

Task 7 inventory lists checkpoint filenames only; `.pt` tensor contents were not inspected or modified.

| Checkpoint | Current provenance status |
| ---------- | ------------------------- |
| `checkpoints/coop_prompt_learner_10ep_lr2e-3.pt` | PoorFrogs-local checkpoint; metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/coop_training_vitb32_best_epoch13_f1_2573.pt` | PoorFrogs-local CoOp Phase 1.1 documented baseline checkpoint. |
| `checkpoints/coop_training_vitb32_last_epoch15.pt` | PoorFrogs-local CoOp Phase 1.1 last-epoch checkpoint. |
| `checkpoints/maple_full_prompt_learner_best.pt` | PoorFrogs-local MaPLe checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_full_prompt_learner.pt` | PoorFrogs-local MaPLe checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_full_smoke/maple_full_prompt_learner_best.pt` | PoorFrogs-local smoke-output checkpoint; not a canonical metric result. |
| `checkpoints/maple_full_smoke/maple_full_prompt_learner.pt` | PoorFrogs-local smoke-output checkpoint; not a canonical metric result. |
| `checkpoints/maple_lora_r4_last6_best.pt` | PoorFrogs-local MaPLe+LoRA checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_lora_r4_last6_e3_lr1e-3_best.pt` | PoorFrogs-local MaPLe+LoRA checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_lora_r4_last6_e3_lr1e-3.pt` | PoorFrogs-local MaPLe+LoRA checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_lora_r4_last6.pt` | PoorFrogs-local MaPLe+LoRA checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_lora_r8_last6_best.pt` | PoorFrogs-local MaPLe+LoRA checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |
| `checkpoints/maple_prompt_learner_best.pt` | PoorFrogs-local MaPLe-style checkpoint; reproduced metrics unknown/unverified in Task 7 evidence. |

`.DS_Store` may exist under `checkpoints/`; it is an OS metadata file, not a model checkpoint.

## PoorFrogs-reproduced metrics

Task 7 did not run new experiments. Tasks 1-6 evidence establishes split access, `IWildCamVal` support, train-split class priors, and logit-adjustment selection guardrails, but it does not contain new MaPLe or MaPLe+LoRA metric tables. Therefore MaPLe/MaPLe+LoRA Top-1 and macro-F1 values are unknown/unverified until re-evaluated in PoorFrogs and recorded with evidence.

## CoOp Phase 1.1 baseline documented in PoorFrogs

Phase 1.1 trains CoOp prompt vectors on the local OpenAI CLIP `ViT-B/32` backbone and selects the best checkpoint by validation F1. This section preserves the existing PoorFrogs-local documented baseline and checkpoint provenance; Task 7 did not newly rerun this experiment.

### Training setup

| Setting | Value |
| ------- | ----- |
| Dataset | IWildCam via WILDS |
| Backbone | OpenAI CLIP `ViT-B/32` |
| Method | CoOp prompt learning |
| Context tokens | `n_ctx=16` |
| Context init | `a photo of a` |
| Epochs | 15 |
| Learning rate | `0.002` |
| Weight decay | `1e-5` |
| Validation split | `IWildCamIDVal` |
| Best-checkpoint metric | `F1-macro_all` |
| Best epoch | 13 |
| Best validation F1 | 0.2573 |

### Final evaluation from best checkpoint

`IWildCamOOD` in this table is final-test-only. It must not be used for checkpoint, tau, lambda, learning-rate, epoch, model, or hyperparameter selection.

| Split | Top-1 | F1-macro |
| ----- | ----- | -------- |
| IWildCamIDVal | 71.33% | 25.73% |
| IWildCamID | 63.05% | 25.90% |
| IWildCamOOD | 65.24% | 18.63% |

### Local artifacts

The canonical Phase 1.1 baseline checkpoint is the best checkpoint selected at epoch 13:

| Artifact | Source file from Kaggle output | Meaning |
| -------- | ------------------------------ | ------- |
| `checkpoints/coop_training_vitb32_best_epoch13_f1_2573.pt` | `coop_prompt_learner_best.pt` | Canonical Phase 1.1 baseline; use for reproduction and Phase 2 comparisons |
| `checkpoints/coop_training_vitb32_last_epoch15.pt` | `coop_prompt_learner.pt` | Last-epoch checkpoint; keep only for late-training or overfit analysis |

### Reproduce final evaluation

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_coop.py \
    --model=ViT-B/32 \
    --train-dataset=IWildCam \
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
    --data-location=./data \
    --batch-size=32 \
    --workers=4 \
    --n-ctx=16 \
    --ctx-init="a photo of a" \
    --epochs=0 \
    --load=./checkpoints/coop_training_vitb32_best_epoch13_f1_2573.pt
```

## Corrected OOD comparison criteria

For corrected OOD-generalization studies, select models and hyperparameters using `IWildCamVal` only. Compare final reports against the CoOp Phase 1.1 baseline above using:

1. Selection metric: `IWildCamVal F1-macro_all`
2. Final-report metric: `IWildCamOOD F1-macro_all` after selection is complete
3. Diagnostic context: Top-1 accuracy on `IWildCamVal`, `IWildCamIDVal`, and final-test-only `IWildCamOOD`

`IWildCamIDVal` remains useful for historical ID-focused comparisons, but it is not the corrected OOD-selection split. `IWildCamOOD` is final-test-only and must not participate in model selection or tuning.

Do not treat a Phase 2 run as better if it only improves Top-1 while reducing macro F1, because IWildCam is class-imbalanced and macro F1 better reflects rare-class behavior.

## MaPLe

MaPLe adds deep coupled prompts at every transformer block (text + vision) via a vendored MaPLe CLIP backend. The pipeline smoke test passed locally (1 train batch + 1 eval batch, no crash). Ready for Kaggle training.

Current Task 7 status: PoorFrogs has MaPLe and MaPLe+LoRA checkpoints in `checkpoints/`, but Tasks 1-6 evidence does not provide reproduced MaPLe/MaPLe+LoRA metric tables. Treat MaPLe/MaPLe+LoRA metrics as unknown/unverified until rerun and recorded in PoorFrogs evidence. Historical metrics from `/Users/jky/Downloads/FLYP/FLYP` are reference-only.

### Planned training setup

Same environment as Phase 1.1 CoOp (Kaggle Tesla P100 16GB).

| Setting | Value |
| ------- | ----- |
| Dataset | IWildCam via WILDS |
| Backbone | OpenAI CLIP `ViT-B/32` |
| Method | MaPLe (deep coupled prompts) |
| Context tokens | `n_ctx=2` per block |
| Prompt depth | `maple-prompt-depth=9` (all ViT-B/32 transformer blocks) |
| Epochs | 9 |
| Learning rate | `0.002` |
| Weight decay | `1e-5` |
| Validation split | Historical text used `IWildCamIDVal`; corrected OOD studies should use `IWildCamVal` |
| Best-checkpoint metric | `F1-macro_all` |
| Kaggle mode | `--mode=full_maple` |

### Planned canonical artifacts

| Artifact | Meaning |
| -------- | ------- |
| `checkpoints/maple_full_prompt_learner_best.pt` | MaPLe checkpoint inventory item; selection provenance/metrics unknown-unverified for corrected OOD protocol |
| `checkpoints/maple_full_prompt_learner.pt` | Last-epoch MaPLe checkpoint |

### Kaggle launch

```bash
# Push to GitHub, then from Kaggle notebook:
# TRAIN_METHOD=full_maple in kernel-metadata.json or pass --mode=full_maple
# The kaggle_main.py entrypoint dispatches to src.train_maple_full automatically.
```

## Historical/reference-only metrics

Any MaPLe, MaPLe+LoRA, tau-sweep, A1, or C1 metric inherited from old notes or `/Users/jky/Downloads/FLYP/FLYP` is reference-only until reproduced in `/Users/jky/Downloads/FLYP/PoorFrogs`. Do not use reference-only numbers to claim PoorFrogs completion, checkpoint superiority, or OOD improvement.
