import json

import pytest
import torch

from src.models.loo_bcpd import sequence_groups, shuffled_sequence_groups
from src.models.stp_oracle_audit import (
    build_oracle_audit_report,
    method_selection_oracle,
    sequence_candidate_oracle,
    write_oracle_audit_artifacts,
)


def test_method_selection_oracle_only_recovers_available_correct_predictions():
    labels = torch.tensor([0, 1, 2, 1])
    stp = torch.tensor([1, 0, 0, 1])
    tpa = torch.tensor([0, 0, 2, 2])
    loo = torch.tensor([2, 1, 1, 0])

    result = method_selection_oracle(labels, stp, (tpa, loo))

    assert result.tolist() == [0, 1, 2, 1]


def test_sequence_candidate_oracle_requires_true_class_in_real_sequence_candidates():
    labels = torch.tensor([1, 2, 2, 0])
    fallback = torch.tensor([0, 0, 1, 1])
    tpa = torch.tensor([1, 2, 1, 1])
    groups = ((0, 1), (2, 3))

    result = sequence_candidate_oracle(labels, fallback, tpa, groups)

    assert result.tolist() == [1, 2, 1, 1]


def test_shuffle_is_deterministic_and_preserves_location_and_group_lengths():
    metadata = [
        torch.tensor([0, 10]), torch.tensor([0, 10]),
        torch.tensor([1, 10]), torch.tensor([1, 10]),
        torch.tensor([2, 20]), torch.tensor([2, 20]),
        torch.tensor([3, 20]), torch.tensor([3, 20]),
    ]
    original_lengths = sorted(len(group) for group in sequence_groups(metadata, 0))

    first = shuffled_sequence_groups(metadata, 0, 1, None, 20260720)
    second = shuffled_sequence_groups(metadata, 0, 1, None, 20260720)

    assert first == second
    assert sorted(len(group) for group in first.groups) == original_lengths
    for group in first.groups:
        assert len({int(metadata[index][1].item()) for index in group}) == 1


def test_report_applies_frozen_gate_and_marks_oracles_non_deployable():
    labels = torch.tensor([0, 1, 0, 1])
    tpa = torch.tensor([[0.0, 3.0], [0.0, 3.0], [3.0, 0.0], [3.0, 0.0]])
    stp = torch.tensor([[0.0, 3.0], [0.0, 3.0], [3.0, 0.0], [3.0, 0.0]])
    loo = torch.tensor([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0], [0.0, 3.0]])
    metadata = [torch.tensor([0, 10]), torch.tensor([0, 10]), torch.tensor([1, 20]), torch.tensor([1, 20])]

    report = build_oracle_audit_report(
        labels=labels,
        tpa_logits=tpa,
        stp_mean_logits=stp,
        stp_loo_logits=loo,
        metadata=metadata,
        location_keys=("10", "10", "20", "20"),
        sequence_field_index=0,
        location_field_index=1,
        train_class_counts=torch.tensor([100, 10]),
        bootstrap_samples=16,
        bootstrap_seed=7,
        shuffle_seeds=(11, 12),
    )

    assert report["deployable_method_result"] is False
    assert report["decision"]["outcome"] == "targeted_method_only"
    assert report["oracles"]["method_selection"]["introduced_error_count"] == 0
    assert set(report["strata_method_selection_headroom"]) == {
        "burst_length", "sequence_label_taxonomy", "true_class_frequency", "tpa_confidence_quartile"
    }


def test_artifact_writer_records_firewall_and_refuses_overwrite(tmp_path):
    preregistration = tmp_path / "frozen.json"
    preregistration.write_text(json.dumps({"audit": "STP Oracle Audit v0"}), encoding="utf-8")
    output_dir = tmp_path / "output"
    report = {
        "reference": {"macro_f1": 0.4},
        "oracles": {
            "method_selection": {"macro_f1_headroom": 0.02},
            "sequence_candidate": {"macro_f1_headroom": 0.03},
            "event_constant": {"macro_f1_headroom": 0.04},
        },
        "negative_control": {"real_stp_minus_shuffle_median": 0.01},
        "decision": {"outcome": "stop_new_sequence_aggregators"},
    }

    write_oracle_audit_artifacts(
        output_dir,
        preregistration_path=preregistration,
        manifest={"confirmation_performance_materialized": False},
        class_mapping={"empty_class_index": 0},
        report=report,
    )
    receipt = json.loads((output_dir / "oracle_audit_receipt.json").read_text(encoding="utf-8"))

    assert receipt["val_confirm_predictions_materialized"] is False
    assert receipt["iwildcam_ood_predictions_materialized"] is False
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_oracle_audit_artifacts(
            output_dir,
            preregistration_path=preregistration,
            manifest={},
            class_mapping={},
            report=report,
        )
