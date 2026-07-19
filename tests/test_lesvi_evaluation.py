import math

import pytest
import torch

from src.models.lesvi_evaluation import (
    build_confirmation_report,
    build_metric_support,
    c2w_metrics,
    paired_location_bootstrap,
    stable_calibration_report,
)


def _metadata():
    return [torch.tensor([1, 10]), torch.tensor([1, 10]), torch.tensor([2, 20]), torch.tensor([2, 20]), torch.tensor([3, 30])]


def test_per_estimand_support_is_fixed_and_rotation_is_frozen_before_predictions():
    labels = torch.tensor([0, 1, 2, 2, 3])
    rotation = torch.tensor([True, True, False, False, False])
    support = build_metric_support(labels, _metadata(), sequence_field_index=1, rotation_common_mask=rotation)
    assert support.class_ids["overall"] == (0, 1, 2, 3)
    assert support.class_ids["animal_true_nonempty"] == (1, 2, 3)
    assert support.class_ids["mixed_empty"] == (0, 1)
    assert support.class_ids["one_species_no_empty"] == (2, 3)
    assert support.class_ids["rotation_common"] == (0, 1)


def test_c2w_uses_tpa_correct_denominator_and_zero_denominator_is_nan():
    labels = torch.tensor([0, 1, 2])
    tpa = torch.tensor([[5.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 5.0, 1.0]])
    candidate = torch.tensor([[0.0, 5.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    result = c2w_metrics(labels, tpa, candidate, torch.ones(3, dtype=torch.bool))
    assert result == {"count": 1, "reference_correct_count": 2, "rate": 0.5}
    empty = c2w_metrics(labels, tpa, candidate, torch.tensor([False, False, True]))
    assert empty["reference_correct_count"] == 0
    assert math.isnan(empty["rate"])


def test_location_bootstrap_preserves_repeated_cluster_multiplicity(monkeypatch):
    labels = torch.tensor([0, 1, 0])
    reference = torch.tensor([0, 0, 0])
    candidate = torch.tensor([0, 1, 1])
    calls = []

    original = torch.randint

    def repeated_location(*args, **kwargs):
        result = torch.tensor([0, 0])
        calls.append(result.tolist())
        return result

    monkeypatch.setattr(torch, "randint", repeated_location)
    result = paired_location_bootstrap(labels, reference, candidate, ["a", "a", "b"], [0, 1], samples=1)
    monkeypatch.setattr(torch, "randint", original)
    assert calls == [[0, 0]]
    assert result.samples == 1
    assert result.location_count == 2


def test_rotation_support_shape_must_match_labels():
    with pytest.raises(ValueError, match="Rotation common support"):
        build_metric_support(torch.tensor([0, 1]), _metadata()[:2], sequence_field_index=1, rotation_common_mask=torch.tensor([True]))


def test_diagnostic_outputs_cannot_change_promotion_checks():
    labels = torch.tensor([0, 1, 2, 2, 3])
    rotation_mask = torch.ones(5, dtype=torch.bool)
    support = build_metric_support(labels, _metadata(), sequence_field_index=1, rotation_common_mask=rotation_mask)
    tpa = torch.nn.functional.one_hot(labels, num_classes=4).float() * 2.0
    stp = tpa.clone()
    lesvi = tpa.clone()
    no_visibility = tpa.clone()
    rotations = [tpa.clone() for _ in range(99)]
    common = dict(
        labels=labels,
        support=support,
        location_keys=["a", "a", "b", "b", "c"],
        rotation_logits=rotations,
        rotation_common_mask=rotation_mask,
        context_eligible_mask=rotation_mask,
        rotation_fallback_counts=[0] * 99,
    )
    first = build_confirmation_report(
        logits_by_name={"tpa": tpa, "stp_mean": stp, "lesvi": lesvi, "no_visibility": no_visibility, "vfep_v0_diagnostic": tpa},
        **common,
    )
    second = build_confirmation_report(
        logits_by_name={"tpa": tpa, "stp_mean": stp, "lesvi": lesvi, "no_visibility": no_visibility, "vfep_v0_diagnostic": -tpa},
        **common,
    )
    assert first["promotion"] == second["promotion"]


def test_calibration_report_is_stable_under_input_reordering_with_frame_id_ties():
    logits = torch.tensor([[1.0, 1.0], [1.0, 1.0], [3.0, 0.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1, 0, 1])
    frame_ids = ["b", "a", "d", "c"]
    first = stable_calibration_report(logits, labels, frame_ids, bins=2)
    order = torch.tensor([1, 0, 3, 2])
    second = stable_calibration_report(logits[order], labels[order], [frame_ids[index] for index in order], bins=2)
    assert first == second
    assert first["role"] == "limitation_wording_only"
