import torch


def test_stp_diagnostics_reports_sequence_corrections_and_cluster_bootstrap():
    from src.models.stp_diagnostics import build_stp_diagnostics

    labels = torch.tensor([0, 0, 1, 1])
    frame_logits = torch.tensor([
        [3.0, 0.0],
        [0.0, 3.0],
        [0.0, 3.0],
        [3.0, 0.0],
    ])
    stp_logits = torch.tensor([
        [3.0, 0.0],
        [2.0, 1.0],
        [0.0, 3.0],
        [1.0, 2.0],
    ])
    metadata = [torch.tensor([10]), torch.tensor([10]), torch.tensor([20]), torch.tensor([20])]

    diagnostics = build_stp_diagnostics(
        labels=labels,
        frame_logits=frame_logits,
        stp_logits=stp_logits,
        train_class_counts=torch.tensor([10, 100]),
        metadata=metadata,
        sequence_field_index=0,
        bootstrap_samples=100,
        seed=7,
    )

    assert diagnostics.frame_macro_f1 == 0.5
    assert diagnostics.stp_macro_f1 == 1.0
    assert diagnostics.frame_wrong_stp_correct == 2
    assert diagnostics.frame_correct_stp_wrong == 0
    assert diagnostics.sequence_count == 2
    assert diagnostics.bootstrap.low <= diagnostics.bootstrap.delta <= diagnostics.bootstrap.high
    assert "## Tail Classes" in diagnostics.to_markdown("IWildCamOOD")


def test_stp_diagnostics_sequence_lengths_group_only_matching_metadata():
    from src.models.stp_diagnostics import build_stp_diagnostics

    labels = torch.tensor([0, 0, 1])
    logits = torch.eye(2)[torch.tensor([0, 0, 1])]
    metadata = [torch.tensor([1]), torch.tensor([1]), torch.tensor([2])]

    diagnostics = build_stp_diagnostics(
        labels=labels,
        frame_logits=logits,
        stp_logits=logits,
        train_class_counts=torch.tensor([5, 500]),
        metadata=metadata,
        sequence_field_index=0,
        bootstrap_samples=20,
        seed=0,
    )

    assert diagnostics.sequence_count == 2
    assert diagnostics.sequence_length_counts == {"singleton": 1, "short_2_4": 2, "long_5_plus": 0}
