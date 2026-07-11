import torch
import pytest


def test_sctr_strength_zero_matches_frame_tpa_logits():
    from src.models.sctr import apply_tail_protective_sctr

    frame_logits = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    metadata = [torch.tensor([7]), torch.tensor([7])]

    routed = apply_tail_protective_sctr(
        frame_logits,
        metadata,
        sequence_field_index=0,
        class_priors=torch.tensor([0.1, 0.9]),
        routing_strength=0.0,
        tail_protection_power=1.0,
    )

    assert torch.equal(routed, frame_logits)


def test_sctr_rejects_invalid_strength_even_without_sequence_metadata():
    from src.models.sctr import apply_tail_protective_sctr

    with pytest.raises(ValueError, match="routing strength"):
        apply_tail_protective_sctr(
            torch.tensor([[2.0, 1.0]]),
            metadata=[],
            sequence_field_index=None,
            class_priors=torch.tensor([0.1, 0.9]),
            routing_strength=1.5,
            tail_protection_power=1.0,
        )


def test_sctr_tail_protection_prevents_sequence_from_overwriting_tail_prediction():
    from src.models.sctr import apply_tail_protective_sctr

    frame_logits = torch.tensor([[2.5, 2.0], [0.0, 5.0]])
    metadata = [torch.tensor([7]), torch.tensor([7])]
    class_priors = torch.tensor([0.001, 1.0])

    unprotected = apply_tail_protective_sctr(
        frame_logits,
        metadata,
        sequence_field_index=0,
        class_priors=class_priors,
        routing_strength=1.0,
        tail_protection_power=0.0,
    )
    protected = apply_tail_protective_sctr(
        frame_logits,
        metadata,
        sequence_field_index=0,
        class_priors=class_priors,
        routing_strength=1.0,
        tail_protection_power=1.0,
    )

    assert unprotected[0].argmax().item() == 1
    assert protected[0].argmax().item() == 0
    assert protected[1].argmax().item() == 1


def test_sctr_candidate_rows_keep_sequence_consensus_disabled():
    from src.eval_tail_cache import make_candidate_rows

    rows = make_candidate_rows(
        prototype_scale_grid=[50.0],
        tau_grid=[0.0],
        tail_gamma_grid=[0.0],
        gate_mode_grid=["none"],
        gate_strength_grid=[0.0],
        sequence_eta_grid=[0.0, 0.5],
        prototype_k_grid=[1],
        concept_beta_grid=[0.5],
        include_concept=False,
        sctr_strength_grid=[0.5],
        sctr_tail_protection_grid=[1.0],
    )

    sctr_row = next(row for row in rows if row["head"] == "sctr")

    assert sctr_row["sequence_eta"] == 0.0
    assert sctr_row["sctr_strength"] == 0.5
    assert sctr_row["sctr_tail_protection"] == 1.0
