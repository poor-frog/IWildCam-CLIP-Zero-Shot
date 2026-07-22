# PGF implementation repair 1: deterministic AMP loss scaling

## Trigger

The first PGF-05 pilot attempt for seed `20260721` reached W&B initialization and model training, then stopped at epoch 1, batch 0 because the unscaled gradient for `module.model.visual.proj` was non-finite. The W&B run is `https://wandb.ai/poorfrogs/PoorFrogs/runs/qlj4f2qb`. The run did not complete an epoch, validation, checkpoint, receipt, or pilot manifest.

## Classification

This is a numerical implementation failure under the frozen AMP precision mode. No pilot metric was observed and no forbidden split was opened. The stage-1 fail action therefore permits an implementation repair and rerun without tuning the method.

## Frozen repair

- Keep AMP autocast enabled.
- Keep the model, loss, optimizer, learning rate, weight decay, scheduler, batch size, epochs, seeds, and validation/WiSE selection protocol unchanged.
- For Paper-Grade Training Foundation v0 only, initialize `torch.amp.GradScaler` at `1.0` and use a growth interval of `2147483647`, making the loss scale static for the planned run length.
- Record this AMP policy in every runtime determinism receipt.
- On any remaining non-finite gradient, fail immediately and print the loss, AMP scale, feature ranges, logit scale, and offending-gradient statistics.
- Do not weaken the zero-skipped-step or no-non-finite-event pilot gates.

The repaired source must be committed and the Kaggle launcher must pin that exact commit before seed `20260721` is rerun.
