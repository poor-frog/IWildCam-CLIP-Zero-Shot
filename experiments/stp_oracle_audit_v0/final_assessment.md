# STP Oracle Audit v0: Final Assessment

## Outcome

The frozen decision is `stop_new_sequence_aggregators`.

The primary method-selection oracle improves STP mean macro-F1 from `42.86%` to `44.19%`, a headroom of `+1.34 pp` with a location-bootstrap 95% confidence interval of `[+0.97, +2.21] pp`. The entire interval is below the preregistered `3 pp` stop threshold. The current search over mean, leave-one-out, attention-like, or confidence-gated sequence aggregation should therefore close.

## What the diagnostic upper bounds mean

The sequence-candidate oracle has `+15.82 pp` headroom and the event-constant oracle has `+38.67 pp` headroom. These are label-aware diagnostic ceilings, not implementable methods. They show that real events contain useful class hypotheses and are usually label-homogeneous, but they do not show that another logit aggregator can identify the correct hypothesis without labels.

The within-location shuffle control reaches median macro-F1 `39.69%`, while real-event STP reaches `42.86%`, a `+3.17 pp` real-grouping advantage. Sequence membership is meaningful; the closed conclusion is specifically that the tested aggregation family has little remaining accessible headroom.

## Claim firewall

This assessment covers only the location-disjoint IWildCam Val-Audit development subset: 10,654 frames, 1,747 sequences, and 20 locations. Val-Confirm performance, IWildCam OOD, and CCT-20 were not materialized.

Val-Confirm viability failed because it contains only four supported tail classes and 62.67% supported-class coverage. It must remain unopened, and this audit cannot support a held-out or OOD performance claim.

## Research action

Keep STP mean as the sequence baseline and stop method churn inside the current aggregation family. Any future method effort must begin under a new preregistration and address a materially different bottleneck, such as frame representation or label-free candidate reliability. The high label-aware ceilings may motivate such a new diagnostic, but they do not override this audit's frozen stop decision.
