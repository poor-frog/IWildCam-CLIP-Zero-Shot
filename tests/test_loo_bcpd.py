import torch
import torch.nn.functional as F


def _inputs():
    torch.manual_seed(17)
    features = F.normalize(torch.randn(4, 5), dim=1)
    prototypes = F.normalize(torch.randn(3, 5), dim=1)
    base_logits = torch.tensor([
        [2.1, 0.2, -0.5],
        [0.3, 1.8, -0.1],
        [-0.2, 0.4, 1.5],
        [1.2, 0.1, 0.3],
    ])
    metadata = [torch.tensor([7]), torch.tensor([7]), torch.tensor([7]), torch.tensor([9])]
    return features, prototypes, base_logits, metadata


def test_loo_bcpd_strength_zero_reuses_cached_tpa_logits():
    from src.models.loo_bcpd import build_loo_bcpd_logits

    features, prototypes, base_logits, metadata = _inputs()
    tpa_logits = base_logits + 50.0 * features @ prototypes.t()

    actual = build_loo_bcpd_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=0,
        prototype_scale=50.0,
        strength=0.0,
        cached_tpa_logits=tpa_logits,
    )

    assert torch.equal(actual.logits, tpa_logits)


def test_loo_bcpd_closed_form_matches_bruteforce_and_is_tangent():
    from src.models.loo_bcpd import build_loo_bcpd_logits, leave_one_out_support

    features, prototypes, base_logits, metadata = _inputs()
    strength = 0.7
    result = build_loo_bcpd_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=0,
        prototype_scale=50.0,
        strength=strength,
    )
    responsibilities = F.softmax(base_logits, dim=1)
    support = leave_one_out_support(features, responsibilities, (0, 1, 2), target_index=0)
    delta = (support.feature_sums - prototypes * (prototypes * support.feature_sums).sum(dim=1, keepdim=True)) / (
        1.0 + support.weights
    ).unsqueeze(1)
    brute_prototypes = F.normalize(prototypes + strength * delta, dim=1)

    assert torch.allclose((prototypes * delta).sum(dim=1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(result.prototype_scores[0], features[0] @ brute_prototypes.t(), atol=1e-6)


def test_unconstrained_mixing_normalizes_the_raw_displaced_prototype():
    from src.models.loo_bcpd import build_loo_bcpd_logits, leave_one_out_support

    features, prototypes, base_logits, metadata = _inputs()
    strength = 0.7
    result = build_loo_bcpd_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=0,
        prototype_scale=50.0,
        strength=strength,
        variant="unconstrained",
    )
    responsibilities = F.softmax(base_logits, dim=1)
    support = leave_one_out_support(features, responsibilities, (0, 1, 2), target_index=0)
    raw_delta = (support.feature_sums - support.weights.unsqueeze(1) * prototypes) / (1.0 + support.weights).unsqueeze(1)
    brute_prototypes = F.normalize(prototypes + strength * raw_delta, dim=1)

    assert torch.allclose(result.prototype_scores[0], features[0] @ brute_prototypes.t(), atol=1e-6)


def test_loo_linear_omits_only_displaced_prototype_normalization():
    from src.models.loo_bcpd import build_loo_linear_logits

    features, prototypes, base_logits, metadata = _inputs()
    result = build_loo_linear_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=0,
        prototype_scale=50.0,
        strength=0.5,
    )

    assert torch.isfinite(result.logits).all()
    assert not torch.equal(result.logits, base_logits + 50.0 * features @ prototypes.t())


def test_loo_support_excludes_mutated_target_feature_and_responsibility():
    from src.models.loo_bcpd import leave_one_out_support

    features, _, base_logits, _ = _inputs()
    responsibilities = F.softmax(base_logits, dim=1)
    original = leave_one_out_support(features, responsibilities, (0, 1, 2), target_index=1)
    changed_features = features.clone()
    changed_features[1] = F.normalize(torch.tensor([3.0, -2.0, 1.0, 4.0, -1.0]), dim=0)
    changed_logits = base_logits.clone()
    changed_logits[1] = torch.tensor([-4.0, 2.0, 7.0])
    changed = leave_one_out_support(
        changed_features,
        F.softmax(changed_logits, dim=1),
        (0, 1, 2),
        target_index=1,
    )

    assert torch.allclose(original.weights, changed.weights, atol=1e-6)
    assert torch.allclose(original.feature_sums, changed.feature_sums, atol=1e-6)


def test_derangement_has_no_fixed_points_and_is_repeatable():
    from src.models.loo_bcpd import deterministic_derangement

    first = deterministic_derangement(7, seed=19)
    second = deterministic_derangement(7, seed=19)

    assert torch.equal(first, second)
    assert torch.equal(first.sort().values, torch.arange(7))
    assert not torch.any(first == torch.arange(7))


def test_sequence_shuffle_is_deterministic_and_preserves_group_sizes():
    from src.models.loo_bcpd import shuffled_sequence_groups

    metadata = [
        torch.tensor([1, 10, 7]), torch.tensor([1, 10, 7]),
        torch.tensor([1, 11, 7]), torch.tensor([1, 11, 7]),
        torch.tensor([1, 12, 21]), torch.tensor([1, 12, 21]),
        torch.tensor([2, 20, 7]), torch.tensor([2, 20, 7]),
    ]
    first = shuffled_sequence_groups(metadata, sequence_field_index=1, location_field_index=0, hour_field_index=2, seed=5)
    second = shuffled_sequence_groups(metadata, sequence_field_index=1, location_field_index=0, hour_field_index=2, seed=5)

    assert first.groups == second.groups
    assert sorted(len(group) for group in first.groups) == [2, 2, 2, 2]
    assert first.changed_frame_fraction > 0.0


def test_missing_sequence_ids_are_not_grouped_together():
    from src.models.loo_bcpd import sequence_groups

    metadata = [None, torch.tensor([1]), torch.tensor([1]), None]

    assert sequence_groups(metadata, sequence_field_index=0, num_examples=4) == ((0,), (1, 2), (3,))


def test_loo_bcpd_can_change_prototype_and_final_class_ranking():
    from src.models.loo_bcpd import build_loo_bcpd_logits

    torch.manual_seed(8)
    features = F.normalize(torch.randn(4, 6), dim=1)
    prototypes = F.normalize(torch.randn(3, 6), dim=1)
    base_logits = torch.zeros(4, 3)
    metadata = [torch.tensor([1]) for _ in range(4)]
    result = build_loo_bcpd_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=0,
        prototype_scale=50.0,
        strength=1.0,
    )
    tpa_scores = features @ prototypes.t()

    assert torch.any(tpa_scores.argmax(dim=1) != result.prototype_scores.argmax(dim=1))
    assert torch.any(tpa_scores.argmax(dim=1) != result.logits.argmax(dim=1))


def test_bopa_centered_prototype_attenuation_is_class_common_affine():
    torch.manual_seed(23)
    prototypes = torch.nn.functional.normalize(torch.randn(4, 6), dim=1)
    centered = prototypes - prototypes.mean(dim=0, keepdim=True)
    basis = torch.linalg.svd(centered, full_matrices=False).Vh.t()
    rank = int(torch.linalg.matrix_rank(centered).item())
    residual = torch.eye(prototypes.shape[1]) - basis[:, :rank] @ basis[:, :rank].t()
    direction = torch.nn.functional.normalize(residual @ torch.randn(6), dim=0)
    query = torch.nn.functional.normalize(torch.randn(6), dim=0)
    strength = 0.6

    transformed_query = query - strength * torch.dot(query, direction) * direction
    transformed_prototypes = prototypes - strength * (prototypes @ direction).unsqueeze(1) * direction
    transformed_scores = torch.nn.functional.normalize(transformed_query, dim=0) @ torch.nn.functional.normalize(
        transformed_prototypes, dim=1
    ).t()
    base_scores = query @ prototypes.t()
    design = torch.stack((base_scores, torch.ones_like(base_scores)), dim=1)
    affine_solution = torch.linalg.lstsq(design, transformed_scores.unsqueeze(1)).solution.squeeze(1)
    reconstructed = design @ affine_solution

    expected_dot = torch.full((4,), torch.dot(prototypes[0], direction))
    assert torch.allclose(prototypes @ direction, expected_dot, atol=1e-5)
    assert torch.allclose(transformed_scores, reconstructed, atol=1e-5)
