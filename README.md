# PoorFrogs

Zero-shot CLIP evaluation on [IWildCam](https://wilds.stanford.edu/datasets/#iwildcam) (WILDS dataset).

## Setup

```bash
./install.sh
source .venv/bin/activate
```

> **macOS OpenMP warning**: If you see `OMP: Error #15` at runtime, set `KMP_DUPLICATE_LIB_OK=TRUE`.

## Data

The WILDS dataset expects `--data-location` to point to the parent directory containing `iwildcam_v2.0/`.

If you already have IWildCam data elsewhere, create a local symlink:

```bash
mkdir -p data
ln -s PATH_TO_YOUR_IWILDCAM_DATASET data/iwildcam_v2.0
```

Then use `--data-location=./data`.

Optional: generate the prompt-caption training CSV used by the original FLYP-style data pipeline:

```bash
python3 ./scripts/prepare_iwildcam.py \
    --metadata ./data/iwildcam_v2.0/metadata.csv \
    --data_dir ./data/iwildcam_v2.0/train \
    --english_label_path ./src/datasets/iwildcam_metadata/labels.csv \
    --save_file ./data/train.csv
```

This creates `data/train.csv`. It is not required for the current zero-shot evaluation or Phase 1 CoOp training, which read IWildCam directly through WILDS.

## Zero-shot Evaluation

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/main.py \
    --model=ViT-B-16 \
    --template=iwildcam_template \
    --train-dataset=IWildCam \
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
    --data-location=./data
```

On Windows PowerShell, set `PYTHONPATH` separately and omit `KMP_DUPLICATE_LIB_OK` unless you hit an OpenMP duplicate runtime error:

```powershell
$env:PYTHONPATH="."
python src/main.py `
    --model=ViT-B-16 `
    --template=iwildcam_template `
    --train-dataset=IWildCam `
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD `
    --data-location=./data
```

### Weights & Biases tracking

Login once:

```bash
wandb login
```

Then add W&B flags to the eval command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/main.py \
    --model=ViT-B-16 \
    --template=iwildcam_template \
    --train-dataset=IWildCam \
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
    --data-location=./data \
    --wandb \
    --wandb-project=PoorFrogs \
    --wandb-run-name=vit-b-16-iwildcam
```

Windows PowerShell version:

```powershell
$env:PYTHONPATH="."
python src/main.py `
    --model=ViT-B-16 `
    --template=iwildcam_template `
    --train-dataset=IWildCam `
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD `
    --data-location=./data `
    --wandb `
    --wandb-project=PoorFrogs `
    --wandb-run-name=vit-b-16-iwildcam
```

W&B logs config (including `device`, `system_hostname`, and `system_user`), per-split metrics, and the final summary table.

## CoOp Prompt Learning

Phase 1 supports CoOp with local OpenAI CLIP models only: `RN50` and `ViT-B/32`. Do not use `ViT-B-16` or `ViT-L-14` for CoOp yet, because those names load `open_clip` models whose text encoder internals differ from the original CoOp implementation.

Small smoke run:

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/train_coop.py \
    --model=ViT-B/32 \
    --train-dataset=IWildCam \
    --eval-datasets=IWildCamIDVal \
    --data-location=./data \
    --batch-size=4 \
    --workers=0 \
    --n-ctx=16 \
    --ctx-init="a photo of a" \
    --epochs=1 \
    --max-train-batches=1 \
    --max-eval-batches=1
```

Full CoOp training example:

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
    --epochs=50 \
    --lr=0.002 \
    --wd=1e-5 \
    --wandb \
    --wandb-project=PoorFrogs \
    --wandb-run-name=coop-vit-b32-iwildcam \
    --save=./checkpoints/coop_prompt_learner.pt
```

Windows PowerShell smoke run:

```powershell
$env:PYTHONPATH="."
python src/train_coop.py `
    --model=ViT-B/32 `
    --train-dataset=IWildCam `
    --eval-datasets=IWildCamIDVal `
    --data-location=./data `
    --batch-size=4 `
    --workers=0 `
    --n-ctx=16 `
    --ctx-init="a photo of a" `
    --epochs=1 `
    --max-train-batches=1 `
    --max-eval-batches=1
```

### Model options

| Model         | Source              | Pretrained       |
|---------------|---------------------|------------------|
| `ViT-B-16`    | open_clip           | LAION-400M       |
| `ViT-L-14`    | open_clip           | LAION-400M       |
| `RN50`        | OpenAI CLIP (local) | OpenAI           |
| `ViT-B/32`    | OpenAI CLIP (local) | OpenAI           |

## Output

```
=== Zero-shot evaluation on: ['IWildCamIDVal', 'IWildCamID', 'IWildCamOOD'] ===

Evaluating on IWildCamIDVal...
  IWildCamIDVal Top-1 accuracy: 0.1066
  IWildCamIDVal F1-macro_all: 0.0887
Evaluating on IWildCamID...
  IWildCamID Top-1 accuracy: 0.1042
  IWildCamID F1-macro_all: 0.0865
Evaluating on IWildCamOOD...
  IWildCamOOD Top-1 accuracy: 0.1289
  IWildCamOOD F1-macro_all: 0.0733

=== Summary ===
| Split         | Top-1  | F1-macro |
| ------------- | ------ | -------- |
| IWildCamIDVal | 10.66% | 8.87%    |
| IWildCamID    | 10.42% | 8.65%    |
| IWildCamOOD   | 12.89% | 7.33%    |
```

## Project Structure

```
src/
├── main.py                      # Entry point (zero-shot only)
├── config.py                    # Argument parser (minimal)
├── datasets/
│   ├── __init__.py              # Exports IWildCam classes only
│   ├── iwildcam.py              # IWildCam dataset class (WILDS wrapper)
│   ├── dataloader.py            # Shared dataloader utilities
│   └── iwildcam_metadata/
│       └── labels.csv           # Class name ↔ label mapping
├── templates/
│   ├── __init__.py              # Exports iwildcam_template only
│   └── iwildcam.py              # Prompt templates
└── models/
    ├── __init__.py
    ├── clip_encoder.py          # CLIPEncoder, ImageEncoder, ClassificationHead, ImageClassifier
    ├── zeroshot.py              # Zero-shot classification head builder
    └── eval.py                  # Evaluation loop
scripts/
└── prepare_iwildcam.py          # Dataset CSV preparation
clip/                            # OpenAI CLIP (local package)
.venv/                           # Virtual environment created by install.sh
```
