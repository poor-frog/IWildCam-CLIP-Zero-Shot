import torch


def test_fixed_class_location_bootstrap_preserves_requested_budget():
    from src.models.vfep_pilot import paired_fixed_class_location_bootstrap

    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    reference = torch.tensor([0, 2, 2, 0, 2, 1])
    candidate = torch.tensor([0, 1, 2, 0, 1, 2])
    result = paired_fixed_class_location_bootstrap(
        labels,
        reference,
        candidate,
        ("a", "a", "a", "b", "b", "b"),
        (0, 1, 2),
        samples=40,
        seed=3,
    )

    assert result.samples == 40
    assert result.location_count == 2
    assert result.delta > 0.0


def test_vfep_pilot_gives_vfep_and_stp_equal_strength_grids():
    from src.models.vfep_pilot import build_vfep_pilot_report

    labels = torch.tensor([1, 1, 1, 2, 2, 2])
    logits = torch.tensor([
        [0.0, 3.0, 0.2],
        [0.0, 0.3, 2.0],
        [0.0, 2.8, 0.1],
        [0.0, 0.1, 2.9],
        [0.0, 1.8, 0.4],
        [0.0, 0.2, 2.5],
    ])
    metadata = [
        torch.tensor([1, 10]), torch.tensor([1, 10]), torch.tensor([1, 10]),
        torch.tensor([2, 20]), torch.tensor([2, 20]), torch.tensor([2, 20]),
    ]
    grid = (0.0, 0.25, 0.5, 1.0)
    report, _ = build_vfep_pilot_report(
        labels=labels,
        tpa_logits=logits,
        metadata=metadata,
        location_keys=("1", "1", "1", "2", "2", "2"),
        sequence_field_index=1,
        location_field_index=0,
        strengths=grid,
        stp_strengths=grid,
        bootstrap_samples=20,
        bootstrap_seed=7,
        shuffle_seeds=(11, 12),
    )

    assert [row["strength"] for row in report["vfep_selection"]] == list(grid)
    assert [row["strength"] for row in report["stp_selection"]] == list(grid)
    assert report["visibility_invariants"]["empty_predictions_equal"] is True
    assert report["visibility_invariants"]["empty_confusion_equal"] is True
    assert report["visibility_invariants"]["empty_to_animal"] == 0
    assert report["visibility_invariants"]["animal_to_empty"] == 0
    assert report["legacy_active_class_location_bootstrap"]["requested_samples"] == 20
    assert report["equal_location_top1_sensitivity"]["location_count"] == 2
    assert len(report["shuffles"]) == 2
    assert "hard_selective_v0" not in report["controls"]


def test_vfep_pilot_marks_an_unavailable_shuffle_control_invalid():
    from src.models.vfep_pilot import build_vfep_pilot_report

    labels = torch.tensor([1, 1, 1, 2, 2])
    logits = torch.tensor([[0.0, 2.0, 0.1], [0.0, 1.9, 0.1], [0.0, 2.1, 0.1], [0.0, 0.1, 2.0], [0.0, 0.1, 1.9]])
    metadata = [*(torch.tensor([1, 10]) for _ in range(3)), *(torch.tensor([1, 20]) for _ in range(2))]

    report, _ = build_vfep_pilot_report(
        labels=labels,
        tpa_logits=logits,
        metadata=metadata,
        location_keys=("1",) * 5,
        sequence_field_index=1,
        location_field_index=0,
        strengths=(0.0,),
        stp_strengths=(0.0,),
        bootstrap_samples=4,
        bootstrap_seed=7,
        shuffle_seeds=(11,),
    )

    assert report["shuffle_control_valid"] is False
    assert report["real_minus_shuffle_median"] is None
