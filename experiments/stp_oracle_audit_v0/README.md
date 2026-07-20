# STP Oracle Audit v0

This is a development-only error-decomposition audit for the frozen DRM-WiSE-TPA-STP stack. It measures whether better sequence decisions still have meaningful headroom; none of its label-aware oracle results are deployable method results.

The authoritative protocol is [`preregistration.json`](preregistration.json). The audit is locked to the location-disjoint `IWildCam Val-Audit` subset, DRM WiSE alpha `0.2`, TPA scale `50`, `K=1`, and STP eta `0.5`. Val-Confirm, ID, OOD, and CCT-20 are forbidden.

## Task ledger

| ID | Task | Status | Artifact |
| --- | --- | --- | --- |
| OA-00 | Freeze audit definitions, firewalls, uncertainty, and decision gate | done | `experiments/stp_oracle_audit_v0/preregistration.json` |
| OA-01 | Implement three diagnostic oracles, within-location shuffle control, strata, and immutable receipt | done | `src/models/stp_oracle_audit.py` |
| OA-02 | Integrate Val-Audit-only evaluator and Kaggle launcher | done | `src/eval_tail_cache.py`, `kaggle-stp-oracle-audit-v0/` |
| OA-03 | Validate compile, unit boundaries, package contract, and synthetic artifact generation | done | `tests/test_stp_oracle_audit.py`, `tests/test_kaggle_stp_oracle_audit_package.py` |
| OA-04 | Execute the frozen full-data Kaggle audit and archive downloaded output | todo | `outputs_log/kaggle-stp-oracle-audit-v0/` |

## Frozen interpretation

- Method-selection oracle headroom below `3 pp`: stop developing new sequence aggregators.
- Headroom at or above `5 pp`: permit exactly one failure-targeted method derived from the preregistered strata.
- Headroom from `3 pp` to below `5 pp`: inconclusive; do not claim a method direction from this audit alone.

The real-event minus shuffled-event result diagnoses whether sequence membership itself carries useful signal. It does not override the primary headroom gate.

## Verification gates

- The output directory must be empty; the writer refuses overwrite.
- The receipt must state that Val-Confirm and OOD predictions were not materialized.
- The output must bind the preregistration, manifest, class mapping, JSON report, and Markdown report by SHA-256.
- A full-data result is not available until OA-04 completes and the downloaded artifact passes the receipt/hash checks.
