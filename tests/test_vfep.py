import torch


def _vfep_inputs():
    logits = torch.tensor([
        [3.0, 2.0, 0.0, -1.0],
        [0.0, 3.0, 0.2, -0.5],
        [0.0, 2.5, 0.4, -0.2],
        [0.3, 0.1, 2.0, -0.4],
        [0.2, 0.0, 2.4, -0.1],
        [2.5, 0.5, 0.2, 0.0],
    ])
    metadata = [
        torch.tensor([1, 10]),
        torch.tensor([1, 10]),
        torch.tensor([1, 10]),
        torch.tensor([2, 20]),
        torch.tensor([2, 20]),
        torch.tensor([2, 20]),
    ]
    return logits, metadata


def test_vfep_preserves_visibility_posterior_and_empty_prediction():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    result = build_vfep_logits(logits, metadata, sequence_field_index=1, strength=1.0)
    original_probabilities = torch.softmax(logits, dim=1)
    vfep_probabilities = torch.softmax(result.logits, dim=1)

    assert torch.allclose(original_probabilities[:, 0], vfep_probabilities[:, 0], atol=1e-6)
    assert torch.equal(logits.argmax(dim=1).eq(0), result.logits.argmax(dim=1).eq(0))
    assert torch.allclose(vfep_probabilities[:, 1:].sum(dim=1), original_probabilities[:, 1:].sum(dim=1), atol=1e-6)


def test_vfep_strength_zero_and_singleton_are_exact_noops():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    zero = build_vfep_logits(logits, metadata, sequence_field_index=1, strength=0.0)
    singleton = build_vfep_logits(logits[:1], metadata[:1], sequence_field_index=1, strength=1.0)

    assert torch.equal(zero.logits, logits)
    assert torch.equal(singleton.logits, logits[:1])


def test_vfep_duplicate_support_does_not_change_target_output():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    base = build_vfep_logits(logits[:3], metadata[:3], sequence_field_index=1, strength=0.7)
    duplicated_logits = torch.cat((logits[:3], logits[1:3]), dim=0)
    duplicated_metadata = [*metadata[:3], *metadata[1:3]]
    duplicated = build_vfep_logits(duplicated_logits, duplicated_metadata, sequence_field_index=1, strength=0.7)

    assert torch.allclose(base.logits[0], duplicated.logits[0], atol=1e-6)


def test_vfep_frame_order_invariance():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    order = torch.tensor([2, 0, 1, 5, 3, 4])
    inverse = torch.argsort(order)
    expected = build_vfep_logits(logits, metadata, sequence_field_index=1, strength=0.5)
    actual = build_vfep_logits(logits[order], [metadata[index] for index in order], sequence_field_index=1, strength=0.5)

    assert torch.allclose(expected.logits, actual.logits[inverse], atol=1e-6)


def test_vfep_shuffle_is_deterministic_and_breaks_original_pairs():
    from src.models.vfep import build_vfep_shuffle_groups

    metadata = []
    for sequence in range(4):
        metadata.extend([torch.tensor([1, sequence]), torch.tensor([1, sequence])])
    first = build_vfep_shuffle_groups(metadata, sequence_field_index=1, location_field_index=0, seed=20260718)
    second = build_vfep_shuffle_groups(metadata, sequence_field_index=1, location_field_index=0, seed=20260718)

    assert first.groups == second.groups
    assert sorted(len(group) for group in first.groups) == [2, 2, 2, 2]
    assert first.changed_frame_fraction > 0.0
    assert first.original_pair_retention == 0.0
    assert all(first.available_frame_mask)


def test_vfep_shuffle_falls_back_to_singletons_when_pair_breaking_is_impossible():
    from src.models.vfep import build_vfep_shuffle_groups

    metadata = [*(torch.tensor([1, 10]) for _ in range(3)), *(torch.tensor([1, 20]) for _ in range(2))]
    shuffle = build_vfep_shuffle_groups(metadata, sequence_field_index=1, location_field_index=0, seed=20260718)

    assert shuffle.groups == ((0,), (1,), (2,), (3,), (4,))
    assert shuffle.available_frame_mask == (False, False, False, False, False)
    assert shuffle.original_pair_retention == 0.0


def test_vfep_all_class_control_is_finite():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    result = build_vfep_logits(logits, metadata, sequence_field_index=1, strength=1.0, factorized=False)

    assert torch.isfinite(result.logits).all()


def test_vfep_location_prior_falls_back_for_missing_target_sequence():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    metadata[0] = torch.tensor([1, -1])
    result = build_vfep_logits(
        logits,
        metadata,
        sequence_field_index=1,
        location_field_index=0,
        support_scope="location",
        strength=1.0,
    )

    assert torch.equal(result.logits[0], logits[0])


def test_vfep_leave_one_out_support_is_independent_of_target_logits():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    original = build_vfep_logits(logits[:3], metadata[:3], sequence_field_index=1, strength=1.0)
    changed_logits = logits[:3].clone()
    changed_logits[0] = torch.tensor([-3.0, -1.0, 5.0, 2.0])
    changed = build_vfep_logits(changed_logits, metadata[:3], sequence_field_index=1, strength=1.0)

    assert original.support_weight[0] == changed.support_weight[0]
    assert original.event_reliability[0] == changed.event_reliability[0]
    assert original.support_count[0] == changed.support_count[0]


def test_vfep_float64_matches_float32_within_tolerance():
    from src.models.vfep import build_vfep_logits

    logits, metadata = _vfep_inputs()
    float32 = build_vfep_logits(logits, metadata, sequence_field_index=1, strength=0.75)
    float64 = build_vfep_logits(logits.double(), metadata, sequence_field_index=1, strength=0.75)

    assert float64.logits.dtype == torch.float64
    assert torch.allclose(float32.logits.double(), float64.logits, atol=1e-6)
