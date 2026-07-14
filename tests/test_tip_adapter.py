import torch
from types import SimpleNamespace

import pytest


def test_tip_adapter_cache_logits_match_dense_one_hot_reference():
    from src.models.tip_adapter import tip_adapter_cache_logits

    query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    cache_features = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    cache_labels = torch.tensor([0, 0, 1])
    beta = 2.0

    actual = tip_adapter_cache_logits(
        query_features,
        cache_features,
        cache_labels,
        num_classes=2,
        beta=beta,
        query_chunk_size=1,
        cache_chunk_size=2,
        device="cpu",
    )

    normalized_query = torch.nn.functional.normalize(query_features, dim=-1)
    normalized_cache = torch.nn.functional.normalize(cache_features, dim=-1)
    affinity = torch.exp(-beta * (1.0 - normalized_query @ normalized_cache.t()))
    reference = affinity @ torch.nn.functional.one_hot(cache_labels, num_classes=2).float()

    assert torch.allclose(actual, reference)


def test_tip_adapter_cache_logits_are_invariant_to_chunk_sizes():
    from src.models.tip_adapter import tip_adapter_cache_logits

    query_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cache_features = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])
    cache_labels = torch.tensor([0, 0, 1, 1])

    small_chunks = tip_adapter_cache_logits(
        query_features,
        cache_features,
        cache_labels,
        num_classes=2,
        beta=1.5,
        query_chunk_size=1,
        cache_chunk_size=1,
        device="cpu",
    )
    large_chunks = tip_adapter_cache_logits(
        query_features,
        cache_features,
        cache_labels,
        num_classes=2,
        beta=1.5,
        query_chunk_size=16,
        cache_chunk_size=16,
        device="cpu",
    )

    assert torch.allclose(small_chunks, large_chunks)


def test_tip_adapter_candidate_rows_include_alpha_beta_grid():
    from src.eval_tail_cache import make_candidate_rows

    rows = make_candidate_rows(
        prototype_scale_grid=[50.0],
        tau_grid=[0.0],
        tail_gamma_grid=[0.0],
        gate_mode_grid=["none"],
        gate_strength_grid=[0.0],
        sequence_eta_grid=[0.0],
        prototype_k_grid=[1],
        concept_beta_grid=[0.5],
        include_concept=False,
        tip_beta_grid=[1.0, 2.0],
        tip_alpha_grid=[0.5],
    )

    tip_rows = [row for row in rows if row["head"] == "tip_adapter"]

    assert [(row["tip_beta"], row["tip_alpha"]) for row in tip_rows] == [(1.0, 0.5), (2.0, 0.5)]


def test_tip_adapter_candidate_rows_include_balanced_support_shot_grid():
    from src.eval_tail_cache import make_candidate_rows

    rows = make_candidate_rows(
        prototype_scale_grid=[0.0],
        tau_grid=[0.0],
        tail_gamma_grid=[0.0],
        gate_mode_grid=["none"],
        gate_strength_grid=[0.0],
        sequence_eta_grid=[0.0],
        prototype_k_grid=[1],
        concept_beta_grid=[0.5],
        include_concept=False,
        tip_beta_grid=[7.0],
        tip_alpha_grid=[0.001],
        tip_support_shots_grid=[1, 4],
    )

    tip_rows = [row for row in rows if row["head"] == "tip_adapter"]

    assert [(row["tip_support_shots"], row["tip_beta"], row["tip_alpha"]) for row in tip_rows] == [
        (1, 7.0, 0.001),
        (4, 7.0, 0.001),
    ]


def test_non_tip_candidate_rows_preserve_the_existing_schema():
    from src.eval_tail_cache import make_candidate_rows

    rows = make_candidate_rows(
        prototype_scale_grid=[50.0],
        tau_grid=[0.0],
        tail_gamma_grid=[0.0],
        gate_mode_grid=["none"],
        gate_strength_grid=[0.0],
        sequence_eta_grid=[0.0],
        prototype_k_grid=[1],
        concept_beta_grid=[0.5],
        include_concept=False,
    )

    assert all("tip_beta" not in row and "tip_alpha" not in row for row in rows)


def test_non_tip_selection_output_does_not_add_tip_columns(capsys):
    from src.eval_tail_cache import print_selection

    row = {
        "head": "default",
        "prototype_scale": 0.0,
        "tau": 0.0,
        "tail_gamma": 0.0,
        "prototype_k": 1,
        "sequence_eta": 0.0,
        "gate_mode": "none",
        "gate_strength": 0.0,
        "sctr_strength": 0.0,
        "sctr_tail_protection": 0.0,
        "concept_beta": None,
        "score": 0.4,
        "top1": 0.5,
        "F1-macro_all": 0.4,
    }

    print_selection([row], {"default": row})

    assert "Tip beta" not in capsys.readouterr().out


def test_tip_adapter_candidate_adds_scaled_cache_residual():
    from src.eval_tail_cache import build_candidate_predictions

    base_logits = torch.tensor([[1.0, 2.0]])
    cache_logits = {(0, 2.0): torch.tensor([[3.0, 4.0]])}
    row = {"head": "tip_adapter", "tip_support_shots": 0, "tip_beta": 2.0, "tip_alpha": 0.5}

    actual = build_candidate_predictions(
        base_logits,
        prototype_raw_logits_by_k={},
        concept_raw_logits=None,
        class_priors=torch.tensor([0.5, 0.5]),
        tail_weights_by_gamma={},
        row=row,
        tip_cache_logits_by_config=cache_logits,
    )

    assert torch.equal(actual, torch.tensor([[2.5, 4.0]]))


def test_balanced_tip_support_limits_each_class_to_requested_shots():
    from src.eval_tail_cache import select_cache_examples

    features = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])

    selected_features, selected_labels = select_cache_examples(
        features,
        labels,
        num_classes=2,
        max_per_class=2,
        seed=7,
    )

    assert selected_features.shape == (4, 2)
    assert torch.equal(torch.bincount(selected_labels, minlength=2), torch.tensor([2, 2]))


def test_tip_support_grid_skips_shots_unavailable_for_a_tail_class():
    from src.eval_tail_cache import resolve_tip_adapter_support_shots_grid

    labels = torch.tensor([0, 0, 0, 1])

    actual = resolve_tip_adapter_support_shots_grid(labels, num_classes=2, support_shots_grid=[1, 2])

    assert actual == [1]


def test_full_data_tip_adapter_requires_iwildcam_val_and_untruncated_features():
    from src.eval_tail_cache import validate_tip_adapter_protocol

    args = SimpleNamespace(
        val_dataset="IWildCamIDVal",
        max_train_batches=None,
        max_eval_batches=None,
        max_cache_examples_per_class=0,
    )

    with pytest.raises(ValueError, match="IWildCamVal"):
        validate_tip_adapter_protocol(args, [1.0])

    args.val_dataset = "IWildCamVal"
    args.max_train_batches = 1
    with pytest.raises(ValueError, match="max-train-batches"):
        validate_tip_adapter_protocol(args, [1.0])

    args.max_train_batches = None
    args.max_eval_batches = 1
    with pytest.raises(ValueError, match="max-eval-batches"):
        validate_tip_adapter_protocol(args, [1.0])
