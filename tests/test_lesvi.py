import math

import pytest
import torch

from src.models.lesvi import (
    LesviConfigurationError,
    LesviPrior,
    _log_event_likelihoods,
    build_donor_event_rotation,
    build_lesvi_logits,
    estimate_lesvi_prior,
    load_prior_artifact,
    prior_predictive,
    verify_lesvi_synthetic_reference,
    with_visibility_variant,
    write_prior_artifact,
)


def _prior() -> LesviPrior:
    pi = torch.tensor([0.3, 0.35, 0.35], dtype=torch.float64)
    theta = torch.tensor([0.6, 0.8], dtype=torch.float64)
    return LesviPrior(
        pi=pi,
        theta=theta,
        mu=prior_predictive(pi, theta),
        global_visibility=0.7,
        class_mapping_sha256="mapping",
        diagnostics={},
    )


def _metadata(sequence_values, location_values=None):
    location_values = [1] * len(sequence_values) if location_values is None else location_values
    return [torch.tensor([location, sequence]) for sequence, location in zip(sequence_values, location_values, strict=True)]


def test_train_only_prior_uses_valid_single_species_events_and_global_fallback():
    labels = torch.tensor([0, 0, 1, 0, 1, 2, 1, 2])
    metadata = _metadata([1, 1, 2, 2, 2, 3, 4, 4])
    prior = estimate_lesvi_prior(
        labels,
        metadata,
        sequence_field_index=1,
        location_field_index=0,
        class_count=4,
        class_mapping_sha256="x",
    )
    assert prior.theta[0].item() == pytest.approx(3 / 5)
    assert prior.theta[1].item() == pytest.approx(2 / 3)
    assert prior.theta[2].item() == pytest.approx(prior.global_visibility)
    assert prior.diagnostics["excluded_multi_species_event_count"] == 1
    assert prior.mu.sum().item() == pytest.approx(1.0)


def test_train_prior_excludes_sequences_with_missing_or_conflicting_locations():
    labels = torch.tensor([1, 1, 2, 2, 0, 0])
    metadata = _metadata([1, 1, 2, 2, 3, 3], [4, 5, -1, -1, 6, 6])
    prior = estimate_lesvi_prior(
        labels,
        metadata,
        sequence_field_index=1,
        location_field_index=0,
        class_count=3,
        class_mapping_sha256="x",
    )
    assert prior.diagnostics["excluded_invalid_location_event_count"] == 2
    assert prior.diagnostics["valid_empty_event_count"] == 1
    assert prior.diagnostics["valid_single_species_event_count"] == 0


def test_likelihood_ratios_integrate_to_one_under_event_prior():
    logits = torch.tensor([[2.0, 0.5, -0.5], [0.0, 1.0, 0.25]])
    prior = _prior()
    likelihood = _log_event_likelihoods(logits, prior).exp()
    normalized = likelihood @ prior.pi
    torch.testing.assert_close(normalized, torch.ones_like(normalized), atol=1e-10, rtol=1e-10)


def test_closed_form_matches_pseudo_joint_enumeration():
    logits = torch.tensor([[0.2, 1.1, -0.7], [1.0, 0.4, -0.2], [-0.4, 0.3, 1.2]])
    prior = _prior()
    result = build_lesvi_logits(logits, _metadata([5, 5, 5]), sequence_field_index=1, prior=prior)
    b = torch.softmax(logits.double(), dim=1)
    mu = prior.mu
    theta = prior.theta
    psi = torch.zeros(3, 3, dtype=torch.float64)
    psi[0, 0] = 1.0
    psi[1, 0], psi[1, 1] = 1.0 - theta[0], theta[0]
    psi[2, 0], psi[2, 2] = 1.0 - theta[1], theta[1]
    expected = []
    for target in range(3):
        event = prior.pi.clone()
        for context in range(3):
            if context == target:
                continue
            event *= (psi * (b[context] / mu).unsqueeze(0)).sum(dim=1)
        event /= event.sum()
        nu = event @ psi
        posterior = b[target] * nu / mu
        posterior /= posterior.sum()
        expected.append(posterior)
    torch.testing.assert_close(torch.softmax(result.logits.double(), dim=1), torch.stack(expected), atol=1e-7, rtol=1e-7)


def test_no_visibility_exact_boundary_and_direct_logit_equivalence():
    prior = with_visibility_variant(_prior(), "none")
    assert torch.equal(prior.theta, torch.ones_like(prior.theta))
    assert prior.mu[0].item() == pytest.approx(prior.pi[0].item())
    logits = torch.tensor([[0.1, 1.2, -0.4], [0.7, 0.2, 0.1]])
    log_likelihood = _log_event_likelihoods(logits, prior)
    log_b = torch.log_softmax(logits.double(), dim=1)
    torch.testing.assert_close(log_likelihood[:, 1:], log_b[:, 1:] - prior.mu[1:].log(), atol=1e-10, rtol=1e-10)
    assert torch.isfinite(log_likelihood).all()


def test_leave_one_out_support_is_target_independent_and_fallback_is_bitwise():
    logits = torch.tensor([[0.1, 1.0, 0.2], [0.3, 0.5, 0.7], [1.0, -0.2, 0.1]])
    metadata = _metadata([9, 9, 9])
    first = build_lesvi_logits(logits, metadata, sequence_field_index=1, prior=_prior())
    changed = logits.clone()
    changed[0] = torch.tensor([10.0, -10.0, 2.0])
    second = build_lesvi_logits(changed, metadata, sequence_field_index=1, prior=_prior())
    torch.testing.assert_close(first.event_posterior[0], second.event_posterior[0])
    singleton = build_lesvi_logits(logits[:1], _metadata([-1]), sequence_field_index=1, prior=_prior())
    assert torch.equal(singleton.logits, logits[:1])
    assert singleton.metadata_fallback_mask.item()


def test_species_permutation_equivariance_and_contradictory_context_is_finite():
    logits = torch.tensor([[0.0, 8.0, -8.0], [0.0, -8.0, 8.0], [0.2, 0.1, 0.0]])
    metadata = _metadata([1, 1, 1])
    result = build_lesvi_logits(logits, metadata, sequence_field_index=1, prior=_prior())
    permutation = torch.tensor([0, 2, 1])
    base = _prior()
    permuted_prior = LesviPrior(
        pi=base.pi[permutation],
        theta=base.theta.flip(0),
        mu=base.mu[permutation],
        global_visibility=base.global_visibility,
        class_mapping_sha256="mapping",
        diagnostics={},
    )
    permuted = build_lesvi_logits(logits[:, permutation], metadata, sequence_field_index=1, prior=permuted_prior)
    torch.testing.assert_close(permuted.logits, result.logits[:, permutation], atol=1e-6, rtol=1e-6)
    assert torch.isfinite(result.logits).all()


def test_long_event_float32_output_matches_float64_posterior_reference():
    generator = torch.Generator().manual_seed(3)
    logits = torch.randn(200, 3, generator=generator)
    result = build_lesvi_logits(logits.float(), _metadata([4] * 200), sequence_field_index=1, prior=_prior())
    assert result.logits.dtype == torch.float32
    assert torch.isfinite(result.logits).all()
    reference = build_lesvi_logits(logits.double(), _metadata([4] * 200), sequence_field_index=1, prior=_prior())
    torch.testing.assert_close(torch.softmax(result.logits.double(), 1), torch.softmax(reference.logits, 1), atol=2e-6, rtol=2e-6)


def test_nonfinite_baseline_is_confirmation_error():
    with pytest.raises(LesviConfigurationError, match="Non-finite baseline"):
        build_lesvi_logits(torch.tensor([[math.nan, 0.0, 1.0]]), _metadata([1]), sequence_field_index=1, prior=_prior())


def test_empty_and_duplicated_context_have_locked_product_evidence_behavior():
    target = torch.tensor([[0.0, 0.5, 0.4]])
    empty_context = torch.tensor([[8.0, -4.0, -4.0]])
    logits_two = torch.cat((target, empty_context), dim=0)
    logits_three = torch.cat((target, empty_context, empty_context), dim=0)
    two = build_lesvi_logits(logits_two, _metadata([2, 2]), sequence_field_index=1, prior=_prior())
    three = build_lesvi_logits(logits_three, _metadata([2, 2, 2]), sequence_field_index=1, prior=_prior())
    assert three.event_posterior[0, 0] > two.event_posterior[0, 0]
    assert torch.isfinite(three.logits).all()


def test_prior_artifact_rejects_class_mapping_or_predictive_mismatch(tmp_path):
    path = tmp_path / "prior.json"
    write_prior_artifact(path, _prior())
    loaded = load_prior_artifact(path, expected_class_mapping_sha256="mapping")
    torch.testing.assert_close(loaded.mu, _prior().mu)
    with pytest.raises(LesviConfigurationError, match="class mapping"):
        load_prior_artifact(path, expected_class_mapping_sha256="wrong")
    payload = __import__("json").loads(path.read_text())
    payload["mu"][0] += 0.01
    payload["mu"][1] -= 0.01
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(LesviConfigurationError, match="mu does not match"):
        load_prior_artifact(path)


def test_donor_rotation_is_deterministic_derangement_with_full_support():
    metadata = _metadata([1, 1, 2, 2, 3, 3], [7] * 6)
    first = build_donor_event_rotation(metadata, sequence_field_index=1, location_field_index=0, seed=12)
    second = build_donor_event_rotation(metadata, sequence_field_index=1, location_field_index=0, seed=12)
    assert first.support_by_target == second.support_by_target
    assert first.available_mask.all()
    for target, support in enumerate(first.support_by_target):
        assert support
        assert metadata[target][1].item() != metadata[support[0]][1].item()


def test_freeze_synthetic_reference_passes_before_specification():
    report = verify_lesvi_synthetic_reference()
    assert report["passed"] is True
    assert report["pseudo_joint_posterior_max_error"] <= 1e-6
