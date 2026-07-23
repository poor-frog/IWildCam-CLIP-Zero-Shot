# PGF infrastructure repair 2: one Kaggle kernel per pilot seed

## Trigger

Kaggle kernel version 6 used the repaired source commit `ebbd66c6824135b1fa22d76995ecf512bb0cd2bb`. Seed `20260721` completed all 20 epochs, selected WiSE alpha `0.3`, wrote both checkpoints, and wrote its receipt. Seed `20260722` reached epoch 18 before the combined process hit Kaggle's approximately 12-hour runtime boundary. Seed `20260723` did not start. Kaggle reported `CANCEL_ACKNOWLEDGED` and published no output files, so no artifact from that run is admissible.

## Classification

This is an infrastructure scheduling failure. The numerical repair passed its immediate gate: the run contained no non-finite event or traceback. No final split was opened and the interrupted metrics are not used for method selection.

## Frozen repair

- Execute seeds `20260721`, `20260722`, and `20260723` in three private Kaggle kernels.
- Each kernel uses the same source commit, dependency pins, dataset attachment, training configuration, AMP policy, validation trace, and WiSE grid.
- Each kernel runs exactly one preregistered seed and writes two checkpoints, one immutable receipt, and one `single_seed_manifest.json`.
- W&B remains disabled because it is secondary evidence only.
- After all three kernel outputs are downloaded, independently replay checkpoint and receipt hashes.
- Create `pilot_manifest.json` only if the three receipts share source, dataset, protocol, and runtime hashes; contain unique checkpoint hashes; and pass the original AMP, non-finite, validation-trace, WiSE-trace, and split-firewall gates.
- Do not reuse the unexported seed `20260721` artifact from the canceled combined run.
- Do not open final evaluation splits.

This repair changes only job partitioning and artifact aggregation. It does not change the method or any frozen hyperparameter.
