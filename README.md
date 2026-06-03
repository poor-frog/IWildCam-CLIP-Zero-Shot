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
ln -s /Users/path/to/your/iwildcamp data/iwildcam_v2.0
```

Then use `--data-location=./data`.

## Zero-shot Evaluation

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/main.py \
    --model=ViT-B-16 \
    --template=iwildcam_template \
    --train-dataset=IWildCam \
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \
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

W&B logs config (including `device`, `system_hostname`, and `system_user`), per-split metrics, and the final summary table.

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
