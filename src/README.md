# `src/` — Main Package

Zero-shot CLIP evaluation pipeline for IWildCam.

## Modules

| Module | Path | Role |
|--------|------|------|
| `main.py` | `src/main.py` | Entry point — parses args, builds zero-shot classifier, runs eval on all splits |
| `config.py` | `src/config.py` | Argument parser with device auto-detection (`cuda` → `mps` → `cpu`) |
| `datasets/` | `src/datasets/` | IWildCam dataset wrappers + dataloader utilities |
| `models/` | `src/models/` | CLIP encoder, classification head, zero-shot head builder, eval loop |
| `templates/` | `src/templates/` | Prompt templates for zero-shot text embedding |
