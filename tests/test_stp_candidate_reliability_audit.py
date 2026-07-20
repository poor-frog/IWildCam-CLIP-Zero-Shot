import hashlib
import json

import pytest
import torch

from src.models.loo_bcpd import sequence_groups
from src.models.stmp_adapter import apply_sequence_consensus
from src.models.stp_candidate_reliability_audit import (
    FEATURE_NAMES,
    build_candidate_reliability_audit_report,
    build_candidate_rows,
    location_fold_assignments,
    write_candidate_reliability_artifacts,
)


def _synthetic_audit_inputs():
    metadata = []
    labels = []
    logits = []
    for location in range(20):
        for local_frame in range(25):
            sequence = location * 5 + local_frame // 5
            label = (location + local_frame // 5) % 5
            prediction = label if local_frame % 5 != 0 else (label + 1) % 5
            row = torch.zeros(5)
            row[prediction] = 4.0
            row[label] += 2.0
            metadata.append(torch.tensor([sequence, location]))
            labels.append(label)
            logits.append(row)
    tpa_logits = torch.stack(logits)
    stp_logits = apply_sequence_consensus(tpa_logits, metadata, 0, eta=0.5)
    return (
        torch.tensor(labels),
        tpa_logits,
        stp_logits,
        metadata,
        tuple(str(int(row[1].item())) for row in metadata),
        torch.tensor([100, 80, 60, 40, 20]),
    )


def test_fold_assignment_is_deterministic_and_has_four_locations_per_fold():
    locations = tuple(str(index) for index in range(20))

    first = location_fold_assignments(locations, fold_count=5, seed=20260721)
    second = location_fold_assignments(tuple(reversed(locations)), fold_count=5, seed=20260721)

    assert first == second
    assert {fold: list(first.values()).count(fold) for fold in range(5)} == {fold: 4 for fold in range(5)}


def test_candidate_pool_and_features_do_not_depend_on_ground_truth_labels():
    labels, tpa, stp, metadata, _, counts = _synthetic_audit_inputs()
    groups = sequence_groups(metadata, 0)

    first = build_candidate_rows(
        labels=labels,
        tpa_logits=tpa,
        stp_logits=stp,
        groups=groups,
        train_class_counts=counts,
    )
    second = build_candidate_rows(
        labels=(labels + 1) % 5,
        tpa_logits=tpa,
        stp_logits=stp,
        groups=groups,
        train_class_counts=counts,
    )

    for frame_index in first:
        assert [(row[0], row[1]) for row in first[frame_index]] == [(row[0], row[1]) for row in second[frame_index]]
    assert any(
        [row[2] for row in first[frame_index]] != [row[2] for row in second[frame_index]]
        for frame_index in first
    )


def test_full_synthetic_oof_audit_obeys_protocol_and_controls():
    labels, tpa, stp, metadata, locations, counts = _synthetic_audit_inputs()

    report, fold_manifest = build_candidate_reliability_audit_report(
        labels=labels,
        tpa_logits=tpa,
        stp_logits=stp,
        metadata=metadata,
        location_keys=locations,
        sequence_field_index=0,
        location_field_index=1,
        train_class_counts=counts,
        bootstrap_samples=32,
        bootstrap_seed=20260721,
    )

    assert report["status"] == "complete"
    assert fold_manifest["all_folds_viable"] is True
    assert all(record["evaluation_location_count"] == 4 for record in fold_manifest["folds"])
    assert all(record["evaluation_frame_count"] == 100 for record in fold_manifest["folds"])
    assert all(record["descriptive_metrics"]["descriptive_only"] for record in fold_manifest["folds"])
    assert all(set(record["standardized_coefficients"]) == set(FEATURE_NAMES) for record in fold_manifest["folds"])
    assert report["feature_names"] == list(FEATURE_NAMES)
    assert report["oof_prediction_count"] == labels.numel()
    assert len(report["oof_predictions"]) == labels.numel()
    assert len(report["negative_controls"]["within_location_sequence_shuffle"]["records"]) == 20
    assert len(report["negative_controls"]["training_target_permutation"]["records"]) == 20
    assert report["negative_controls"]["all_controls_viable"] is True
    assert report["firewall"] == {
        "val_confirm_predictions_materialized": False,
        "iwildcam_ood_predictions_materialized": False,
        "cct20_opened": False,
    }
    assert set(report["promotion"]["conditions"]) == {
        "macro_f1_gain_at_least_3pp",
        "location_bootstrap_lower_bound_gt_zero",
        "animal_macro_f1_non_regression",
        "tail_macro_f1_within_0p5pp",
        "empty_animal_total_error_non_increase",
        "real_gain_minus_shuffle_median_at_least_1pp",
        "permuted_target_gain_median_at_most_0p5pp",
    }


def test_degenerate_candidate_rows_produce_blocked_report_instead_of_crashing():
    labels, _, _, metadata, locations, counts = _synthetic_audit_inputs()
    tpa = torch.zeros(labels.numel(), 5)
    tpa[:, 0] = 1.0

    report, fold_manifest = build_candidate_reliability_audit_report(
        labels=labels,
        tpa_logits=tpa,
        stp_logits=tpa,
        metadata=metadata,
        location_keys=locations,
        sequence_field_index=0,
        location_field_index=1,
        train_class_counts=counts,
        bootstrap_samples=16,
    )

    assert report["status"] == "blocked_fold_viability"
    assert report["promotion_decision_materialized"] is False
    assert fold_manifest["all_folds_viable"] is False
    assert all("candidate_training_viability_error" in fold for fold in fold_manifest["folds"])


def test_report_rejects_location_keys_that_do_not_match_metadata():
    labels, tpa, stp, metadata, locations, counts = _synthetic_audit_inputs()
    wrong_locations = ("999", *locations[1:])

    with pytest.raises(ValueError, match="must match"):
        build_candidate_reliability_audit_report(
            labels=labels,
            tpa_logits=tpa,
            stp_logits=stp,
            metadata=metadata,
            location_keys=wrong_locations,
            sequence_field_index=0,
            location_field_index=1,
            train_class_counts=counts,
        )


def test_writer_hashes_artifacts_and_refuses_overwrite(tmp_path):
    preregistration = tmp_path / "frozen.json"
    preregistration.write_text(json.dumps({"audit": "STP Candidate Reliability Audit v0"}), encoding="utf-8")
    output = tmp_path / "output"
    report = {"status": "blocked_fold_viability"}

    write_candidate_reliability_artifacts(
        output,
        preregistration_path=preregistration,
        fold_manifest={"all_folds_viable": False},
        class_mapping={"empty_class_index": 0},
        report=report,
    )
    receipt = json.loads((output / "audit_receipt.json").read_text(encoding="utf-8"))
    for filename, expected in receipt["artifact_sha256"].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert receipt["val_confirm_predictions_materialized"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_candidate_reliability_artifacts(
            output,
            preregistration_path=preregistration,
            fold_manifest={},
            class_mapping={},
            report=report,
        )


def test_feature_whitelist_matches_frozen_preregistration():
    preregistration = json.loads(
        open("experiments/stp_candidate_reliability_audit_v0/preregistration.json", encoding="utf-8").read()
    )

    assert list(FEATURE_NAMES) == preregistration["feature_whitelist"]
