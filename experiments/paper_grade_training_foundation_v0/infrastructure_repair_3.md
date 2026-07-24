# PGF infrastructure repair 3: stage-2 completion launcher

## Trigger

The three independent pilot outputs passed the frozen PGF-05 gates and were
sealed in `pilot_manifest.json` with SHA-256
`3fad0cc18b0ff1d369f503cbb6605285daf71fb7ed1bde490b0aaffdd9699a66`.
The frozen source commit already authorizes validation seeds `20260724` and
`20260725` in `src/models/paper_grade_training_foundation.py`, but the
single-seed Kaggle helper introduced for the pilot retained a pilot-only
launcher/verifier allowlist.

Without an infrastructure-only compatibility bridge, a completion kernel
would stop before training even though both completion seeds were frozen in
the original preregistration.

## Classification

This is a stage-transition launcher defect. It does not affect the model,
training command, validation selection, dataset, AMP policy, WiSE grid, or
split firewall.

## Frozen repair

- Keep source commit
  `a97bfa5af010096701fe43a08e0f24678123353b` for all five validation runs.
- Keep the source tree, dataset attachment, dependency pins, training
  configuration, T4 machine shape, and `--no-wandb` policy unchanged.
- In each completion launcher only, expand the helper's launcher/verifier
  allowlist to the five seeds already present in the preregistration.
- Run exactly one completion seed per private Kaggle kernel.
- Continue to materialize only checkpoints, the Val trace, the WiSE trace,
  the immutable run receipt, and the single-seed manifest.
- Do not open `IWildCamIDVal`, `IWildCamID`, `IWildCamOOD`, or CCT-20.

This repair changes only the package-level stage-2 dispatch guard. The cloned
training source and the receipt-bound source provenance remain identical to
the three pilot runs.
