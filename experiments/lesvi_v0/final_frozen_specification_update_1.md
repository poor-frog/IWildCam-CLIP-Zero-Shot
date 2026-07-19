# LESVI-v0 Final Frozen Specification — Update 1

## Status

This update preserves the frozen LESVI-v0 method and replaces only the confirmation route.

- The mathematical model, train-only priors, closed-form leave-one-out inference, controls, metrics, rotation test, bootstrap, and promotion gates in `preregistration.json` remain frozen.
- The IWildCam internal-confirmation route is blocked by its preregistered viability gate. `Val-Confirm` has 4,307 frames and 12 locations, but supported-class coverage is 62.67% and only four supported tail classes are present.
- The CCT-20 external-confirmation route is blocked before inference. Its official training annotations contain no valid all-empty event and no single-species event with an observed empty frame, so the train-only visibility prior is not identifiable.
- Neither blocked route is evidence that LESVI succeeds or fails. No LESVI efficacy result, trans-validation result, or trans-test result was opened on either route.

The next objective is therefore not another LESVI variant. It is to identify one dataset and split protocol that can test the already-frozen hypothesis without changing the method after seeing confirmation outcomes.

## Frozen Method Boundary

The following items must not change during this update:

- latent event species \(E_s\), frame nonempty state \(V_i\), and frame label \(Y_i\);
- Laplace-smoothed train-only \(\theta_c\), \(\pi\), and \(\mu\);
- prior-relative likelihood and direct adjusted-logit equations;
- exact leave-one-out context;
- TPA, STP-Mean at `eta=0.5`, `no_visibility`, and 99 donor-event rotations;
- primary fixed-support `F1-macro_all` metric;
- 2,000 paired location-bootstrap replicates with seed `20260718`;
- all promotion thresholds in `preregistration.json`;
- no temperature, strength, event-length normalization, redundancy correction, or validation-selected hyperparameter.

Changing any item above creates a new method version and cannot be reported as LESVI-v0.

## Phase 0 — Close Existing Routes

Record both outcomes as immutable viability failures:

1. `IWildCam-internal`: `blocked_split_viability`.
2. `CCT20-external`: `blocked_train_only_visibility_prior_not_identifiable`.

The closure record must include artifact checksums and state explicitly:

- confirmation predictions opened: `false`;
- trans-validation labels opened for LESVI evaluation: `false`;
- trans-test labels opened: `false`;
- promotion decision: `not_evaluated`.

Do not regenerate either split with a new seed, relax its gate, or estimate priors from validation/test labels.

## Phase 1 — Metadata-Only Candidate Screening

Create a generic preflight that reads annotations and metadata only. It must not load CLIP, images, cached logits, checkpoints, or model predictions.

For every candidate dataset, produce a versioned screening receipt containing:

- annotation source and SHA-256 checksum;
- class mapping with empty fixed at internal index 0;
- split, sequence/event, and location definitions;
- malformed, multi-label, missing-location, and missing-sequence counts;
- valid all-empty event count;
- valid single-species event count;
- species-event count containing both animal and empty frames;
- per-species animal-frame and empty-frame support;
- proposed confirmation frame, class, tail-class, and location coverage computed from aggregate-only annotation access;
- largest-location frame fraction;
- confirmation/test label-access flags, both initially `false`.

A candidate passes the hard identifiability gate only when all are true:

- at least one valid all-empty training event;
- at least one valid single-species training event containing both animal and empty frames;
- global Laplace fallback is estimated exclusively from training events;
- event and location identifiers are available without reading confirmation outcomes;
- the confirmation split has at least 80% supported-class coverage;
- at least five supported tail classes are present under a tail definition frozen from training counts;
- at least ten confirmation locations are present;
- the largest confirmation location contains at most 50% of confirmation frames.

The first two conditions establish structural visibility-prior identifiability. The remaining conditions preserve the already-used confirmation viability standard. Report support counts even when the candidate fails.

Candidate ordering and all annotation sources must be recorded before executing the first screen. A failed candidate cannot be rescued by changing thresholds or category mappings after inspecting its receipt.

## Phase 2 — Select One Confirmation Route

Selection uses metadata receipts only:

1. Reject every candidate that fails a hard gate.
2. If exactly one candidate passes, select it.
3. If multiple candidates pass, select deterministically by:
   - highest supported-class fraction;
   - then highest supported-tail-class count;
   - then highest location count;
   - then lexicographically smallest frozen dataset identifier.
4. If no candidate passes, close LESVI-v0 as `confirmation_unavailable_on_screened_benchmarks`.

No model performance, model confidence, cached logits, or test labels may influence dataset selection. Confirmation annotations may be read only to compute aggregate support and location counts; per-example outcomes and class-composition tables must not be materialized.

## Phase 3 — Dataset Adapter And Freeze

For the selected route, create a dataset-specific adapter without changing LESVI equations. Freeze:

- annotation checksums and snapshot/version;
- class order and normalized names;
- empty-class mapping;
- multi-label exclusion policy;
- train, confirmation, and held-out-test split identifiers;
- sequence/event and location keys;
- prompt templates and frozen TPA construction;
- metric-support construction;
- rotation strata and deterministic seeds;
- source-bundle checksum and immutable genesis ledger.

Then compute train-only priors. Re-run the identifiability gate on the resulting prior artifact before initializing OpenCLIP or loading images. A mismatch between metadata receipt and prior artifact blocks the route.

## Phase 4 — Verification Before Confirmation

The existing LESVI tests remain mandatory, including pseudo-joint enumeration, direct-logit equivalence, exact `no_visibility`, leave-one-out totals, numerical fallbacks, support sets, C2W, rotations, bootstrap multiplicity, and receipt replay protection.

Add route-independent tests for:

- metadata preflight never loading model/image code;
- confirmation and test labels remaining unopened during candidate screening;
- deterministic candidate selection from receipts;
- IWildCam fixture reproducing its split-viability block;
- CCT-20 fixture reproducing its prior-identifiability block;
- a synthetic viable dataset reaching freeze without opening confirmation predictions;
- a failed prior-integrity check preventing model initialization.

The verification artifact must bind test results to the exact source-bundle and frozen-spec checksums.

## Phase 5 — One-Shot Confirmation

Run exactly once on the selected confirmation split and emit:

- TPA;
- STP-Mean `eta=0.5`;
- LESVI-v0;
- `no_visibility`;
- 99 donor-event rotations;
- fixed-support metrics, C2W, location bootstrap, per-location deltas, and leave-one-location-out sensitivity.

Apply the original promotion gates without modification. Diagnostic outputs remain non-promotable and cannot change the method or gates.

If confirmation fails, write an immutable negative receipt and do not open held-out test labels. LESVI-v0 then remains a mechanism-motivated but unconfirmed method.

If confirmation passes, write the positive receipt before any held-out evaluation.

## Phase 6 — Held-Out Evaluation

Open the held-out test split only after a positive confirmation receipt whose checksum matches the frozen source bundle and specification.

Run the same frozen methods and metrics once. Do not select a checkpoint, alpha, strength, prompt, class subset, or reporting slice from held-out results.

The final claim remains limited to an event-local, prior-relative latent correction with soft nonempty visibility and closed-form leave-one-out inference. A positive result does not establish calibrated uncertainty, physical visibility, or general superiority across camera-trap datasets.

## Stop Rules

- No candidate passes metadata preflight: stop LESVI-v0 experiments and report confirmation unavailable.
- Selected candidate fails prior integrity or identifiability: stop before model/image loading.
- Confirmation fails any promotion gate: stop before held-out test.
- Confirmation succeeds: seal the receipt, then permit one held-out evaluation.
- Any proposed method or threshold change: create a separately named successor version and a new preregistration; do not edit LESVI-v0 retroactively.

## Task Ledger

| ID | Task | Owner | Status | Output |
| --- | --- | --- | --- | --- |
| LESVI-001 | Preserve original frozen mathematical specification | research | done | `experiments/lesvi_v0/preregistration.json` |
| LESVI-002 | Close IWildCam internal-confirmation route | research | blocked | `outputs_log/kaggle-flyp-lesvi-freeze/freeze_blocked_receipt.json` |
| LESVI-003 | Close CCT-20 external-confirmation route | research | blocked | `outputs_log/kaggle-cct20-lesvi-freeze/lesvi_cct20_freeze_blocked_receipt.json` |
| LESVI-004 | Implement generic metadata-only candidate preflight | Codex | done | `src/prepare_lesvi_candidate_preflight.py` and registry schemas |
| LESVI-005 | Freeze candidate registry and screen candidates | research | todo | `outputs/lesvi-candidate-screen-v0/` |
| LESVI-006 | Freeze one passing dataset adapter and priors | research | todo | conditional on LESVI-005 |
| LESVI-007 | Run one-shot confirmation | research | todo | conditional on LESVI-006 |
| LESVI-008 | Run held-out evaluation | research | todo | conditional on positive LESVI-007 receipt |

## Immediate Next Action

Freeze the ordered candidate registry and generate normalized metadata inputs for LESVI-005. Do not build another inference launcher or nominate a winning dataset from model performance.
