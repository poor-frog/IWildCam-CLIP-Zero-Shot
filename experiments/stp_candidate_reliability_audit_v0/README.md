# STP Candidate Reliability Audit v0

This audit is a preregistered feasibility test derived from the closed STP Oracle Audit v0. It does not reopen the closed sequence-aggregator search. It asks whether fixed label-free reliability signals can rank the class candidates already present inside a real event.

The authoritative specification is [`preregistration.json`](preregistration.json). Implementation must not change the candidate pool, feature whitelist, selector, cross-validation assignment, controls, metrics, or promotion gate.

## Research question

Can a fixed reliability probe trained on 16 Val-Audit locations select better event candidates on four unseen Val-Audit locations, pooled across five location-grouped folds, without using location identity or ground-truth labels as prediction features?

## Firewall

- Development data: IWildCam Val-Audit only.
- Val-Confirm, IDVal, ID, OOD, and CCT-20 are forbidden.
- Every primary prediction must be out-of-fold by location.
- Oracle labels may define training targets and evaluation metrics only; they may not construct candidates or features.
- This audit cannot produce a deployable-method or held-out-generalization claim.

## Frozen decision

The audit passes only if every promotion condition passes, including at least `+3.0 pp` pooled out-of-fold macro-F1 over STP, a positive location-bootstrap lower bound, no animal macro-F1 regression, bounded tail regression, no increase in empty/animal errors, and successful negative controls.

- Pass: authorize exactly one separately preregistered candidate-reranker experiment.
- Fail: close all further sequence-inference method development on this foundation.
- Inconclusive: repair only the audit implementation and rerun this same specification; do not open a held-out split.

## Task ledger

| ID | Task | Status | Artifact |
| --- | --- | --- | --- |
| CR-00 | Freeze question, feature whitelist, OOF protocol, controls, metrics, and gate | done | `experiments/stp_candidate_reliability_audit_v0/preregistration.json` |
| CR-01 | Implement deterministic fold assignment, candidate rows, and diagnostic selector | done | `src/models/stp_candidate_reliability_audit.py` |
| CR-02 | Add protocol, leakage, control, receipt, and synthetic end-to-end tests | done | `tests/test_stp_candidate_reliability_audit.py` |
| CR-03a | Add a private Val-Audit-only Kaggle launcher | done | `kaggle-stp-candidate-reliability-audit-v0/` |
| CR-03b | Execute the frozen Kaggle audit exactly once and archive output | done | `outputs_log/kaggle-stp-candidate-reliability-audit-v0/` |
| CR-04 | Verify downloaded hashes and apply the frozen promotion gate | done | `experiments/stp_candidate_reliability_audit_v0/closure_receipt.json` |

## Closed outcome

The frozen audit failed its promotion gate and is closed with outcome
`close_all_sequence_inference_development`. See [`final_assessment.md`](final_assessment.md)
and [`closure_receipt.json`](closure_receipt.json). No candidate-reranker experiment
is authorized, and Val-Confirm, IWildCam OOD, and CCT-20 remain unopened.
