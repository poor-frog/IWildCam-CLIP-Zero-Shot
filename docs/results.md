# PoorFrogs Results

This file records reproducible baselines for IWildCam experiments. Use these numbers as the comparison point for later CoOp ablations and Phase 2 backbone work.

## CoOp Phase 1.1 Baseline

Phase 1.1 trains CoOp prompt vectors on the local OpenAI CLIP `ViT-B/32` backbone and selects the best checkpoint by validation F1. The final report below loads the best checkpoint before evaluating all splits.

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

## Phase 2 comparison criteria

For Phase 2, compare against the CoOp Phase 1.1 baseline above using:

1. Primary metric: `IWildCamIDVal F1-macro_all`
2. Secondary metric: `IWildCamOOD F1-macro_all`
3. Tiebreaker: Top-1 accuracy on `IWildCamIDVal` and `IWildCamOOD`

Do not treat a Phase 2 run as better if it only improves Top-1 while reducing macro F1, because IWildCam is class-imbalanced and macro F1 better reflects rare-class behavior.

## MaPLe Phase 2.1a Baseline

Phase 2.1a is the next planned comparison against CoOp Phase 1.1. It uses shallow MaPLe-style multimodal prompting on the same OpenAI CLIP `ViT-B/32` backbone, adding learnable visual prompt tokens while keeping the CLIP backbone frozen.

A Phase 2.1a run should be considered promising only if it improves `IWildCamIDVal F1-macro_all` above 25.73% or `IWildCamOOD F1-macro_all` above 18.63% without a severe Top-1 regression.

Planned canonical checkpoint names:

| Artifact | Meaning |
| -------- | ------- |
| `checkpoints/maple_prompt_learner_best.pt` | Best shallow MaPLe checkpoint selected by `IWildCamIDVal F1-macro_all` |
| `checkpoints/maple_prompt_learner.pt` | Last-epoch shallow MaPLe checkpoint |

## MaPLe

MaPLe adds deep coupled prompts at every transformer block (text + vision) via a vendored MaPLe CLIP backend. The pipeline smoke test passed locally (1 train batch + 1 eval batch, no crash). Ready for Kaggle training.

### Planned training setup

Same environment as Phase 1.1 CoOp (Kaggle Tesla P100 16GB).

| Setting | Value |
| ------- | ----- |
| Dataset | IWildCam via WILDS |
| Backbone | OpenAI CLIP `ViT-B/32` |
| Method | MaPLe (deep coupled prompts) |
| Context tokens | `n_ctx=2` per block |
| Prompt depth | `maple-prompt-depth=9` (all ViT-B/32 transformer blocks) |
| Epochs | 15 |
| Learning rate | `0.002` |
| Weight decay | `1e-5` |
| Validation split | `IWildCamIDVal` |
| Best-checkpoint metric | `F1-macro_all` |
| Kaggle mode | `--mode=full_maple` |

### Planned canonical artifacts

| Artifact | Meaning |
| -------- | ------- |
| `checkpoints/maple_full_prompt_learner_best.pt` | Best MaPLe checkpoint selected by `IWildCamIDVal F1-macro_all` |
| `checkpoints/maple_full_prompt_learner.pt` | Last-epoch MaPLe checkpoint |

### Kaggle launch

```bash
# Push to GitHub, then from Kaggle notebook:
# TRAIN_METHOD=full_maple in kernel-metadata.json or pass --mode=full_maple
# The kaggle_main.py entrypoint dispatches to src.train_maple_full automatically.
```
