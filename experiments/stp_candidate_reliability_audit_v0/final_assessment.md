# STP Candidate Reliability Audit v0: Final Assessment

## Outcome

The frozen decision is `close_all_sequence_inference_development`.

The fixed diagnostic selector decreases pooled out-of-fold macro-F1 from `42.86%` for STP to `41.78%`, a change of `-1.08 pp`. The paired location-bootstrap 95% confidence interval is `[-6.64, +1.93] pp`, so the result neither reaches the preregistered `+3 pp` threshold nor establishes a positive improvement.

Five of the seven required promotion conditions fail. Animal macro-F1 falls from `42.03%` to `41.08%`, tail macro-F1 falls from `27.32%` to `16.54%`, and empty/animal errors increase from `169` to `201`. The selector raises top-1 accuracy from `58.74%` to `63.65%`, but this does not rescue the primary macro-F1 gate and is accompanied by severe tail-class regression.

## What the controls mean

All five location-grouped folds and all 40 negative-control runs are viable. The real selector gain exceeds the within-location sequence-shuffle median by `+6.90 pp`, and the training-target-permutation median gain is `-12.76 pp`. These controls show that the probe learned real sequence-dependent reliability signal rather than succeeding through a broken control.

The learned signal is nevertheless not useful under the frozen objective: it shifts predictions toward higher-accuracy behavior while reducing active-class macro-F1, animal macro-F1, and tail macro-F1. Candidate rankability therefore does not provide a justified path to a deployable reranker on this foundation.

## Claim firewall

This assessment covers only 10,654 pooled out-of-fold predictions from 20 locations in the IWildCam Val-Audit development subset. Every fold met its preregistered viability requirements. Val-Confirm, IWildCam ID, IWildCam OOD, and CCT-20 predictions were not materialized.

The raw artifact hashes match the Kaggle audit receipt, the class mapping contains all 182 classes, and the Kaggle source checkout is commit `538203564d77e23d76e68029f5e7c1b1de12fa9b`. The Kaggle preregistration is semantically identical to the frozen local copy; their byte hashes differ only because the runtime copy was serialized with sorted JSON keys.

## Research action

Do not authorize a candidate-reranker v0 and do not continue with new STP aggregation, attention, gating, or sequence-selection variants on this foundation. Keep STP mean as the current sequence baseline. Any later method effort must address a materially different bottleneck, such as representation or training, under a new preregistration.
