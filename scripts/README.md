# `scripts/` — Utility Scripts

| Script | Role |
|--------|------|
| `prepare_iwildcam.py` | Generates training CSV from raw IWildCam data (not needed for zero-shot eval) |
| `train_coop.sh` | Runs Phase 1/1.1 CoOp prompt learning with OpenAI CLIP `ViT-B/32`, including per-epoch validation and best-checkpoint final eval |
| `train_maple.sh` | Runs MaPLe baseline deep coupled prompt learning with OpenAI CLIP `ViT-B/32`, including per-epoch validation and best-checkpoint final eval |
| `train_maple_lora.sh` | Runs MaPLe + vision-only `out_proj` LoRA with OpenAI CLIP `ViT-B/32`, rank 8, last 6 vision blocks, per-epoch validation, and best-checkpoint final eval |
