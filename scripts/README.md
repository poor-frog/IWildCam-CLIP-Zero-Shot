# `scripts/` — Utility Scripts

| Script | Role |
|--------|------|
| `prepare_iwildcam.py` | Generates training CSV from raw IWildCam data (not needed for zero-shot eval) |
| `train_coop.sh` | Runs Phase 1/1.1 CoOp prompt learning with OpenAI CLIP `ViT-B/32`, per-epoch validation, best-checkpoint final eval |
| `train_maple.sh` | Runs MaPLe baseline deep coupled prompt learning with OpenAI CLIP `ViT-B/16`, batch size 256, per-epoch validation, best-checkpoint final eval |
| `train_maple_lora.sh` | Runs MaPLe + vision-only `out_proj` LoRA with OpenAI CLIP `ViT-B/16`, batch size 256, rank 4, last 6 vision blocks, per-epoch validation, best-checkpoint final eval |
| `train_c1_maple_lora_cbce_kl.sh` | Runs C1: MaPLe + vision-only LoRA + KL-to-zero-shot anchor on OpenAI CLIP `ViT-B/16`, batch size 256, selected by IWildCamVal |
| `colab_train_maple.sh` | Packs the project, uploads to a Colab session, installs deps, runs MaPLe training, downloads checkpoints back |
| `eval_a1_maple_cbce_best_local.sh` | Local eval of A1 (CBCE) best checkpoint on all 4 splits: IWildCamIDVal, IWildCamVal, IWildCamID, IWildCamOOD |
| `eval_maple_tau_sweep_local.sh` | Local vanilla MaPLe tau-sweep eval (logit adjustment) on the same 4 splits with configurable tau grid — no training, epochs=0 |

## Environment variables

| Variable | Used by | Default |
|----------|---------|---------|
| `DATA_LOCATION` | eval scripts | `./data` |
| `CKPT` | eval scripts | varies per script |
| `TAU_GRID` | `eval_maple_tau_sweep_local.sh` | `0,0.25,0.5,0.75,1,1.5,2` |
| `COLAB_SESSION` | `colab_train_maple.sh` | `poorfrogs-maple` |
| `COLAB_GPU` | `colab_train_maple.sh` | `T4` |
| `KL_WEIGHT` | `train_c1_maple_lora_cbce_kl.sh` | `0.1` |
| `KL_TEMPERATURE` | `train_c1_maple_lora_cbce_kl.sh` | `1.0` |
| `SAVE_PATH` | `train_c1_maple_lora_cbce_kl.sh` | `./checkpoints/c1_maple_lora_kl_vitb16_bs256.pt` |
