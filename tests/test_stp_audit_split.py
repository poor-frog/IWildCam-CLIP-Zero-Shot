import hashlib

import pytest
import torch


def _metadata(*rows):
    return [torch.tensor(row) for row in rows]


def test_location_assignment_uses_integer_sha256_threshold():
    from src.models.stp_audit_split import AUDIT_HASH_THRESHOLD, location_split_name

    location = "camera-17"
    digest = hashlib.sha256(f"20260716|{location}".encode("utf-8")).hexdigest()
    expected = "val_audit" if int(digest, 16) < AUDIT_HASH_THRESHOLD else "val_confirm"

    assert location_split_name(location) == expected


def test_location_split_rejects_sequence_with_two_valid_locations():
    from src.models.stp_audit_split import StpAuditSplitError, build_location_audit_split

    metadata = _metadata((1, 7, 8), (2, 7, 8))
    labels = torch.tensor([0, 1])
    train_counts = torch.tensor([30, 10])

    with pytest.raises(StpAuditSplitError, match="multiple valid locations"):
        build_location_audit_split(metadata, labels, train_counts, sequence_field_index=1, location_field_index=0)


def test_location_incomplete_sequence_is_excluded_and_forced_to_tpa_fallback():
    from src.models.stp_audit_split import build_location_audit_split

    metadata = [torch.tensor([1, 7, 8]), torch.tensor([float("nan"), 7, 8]), torch.tensor([2, 9, 8])]
    labels = torch.tensor([0, 1, 1])
    train_counts = torch.tensor([30, 10])

    result = build_location_audit_split(metadata, labels, train_counts, sequence_field_index=1, location_field_index=0)

    assert result.inferential_mask.tolist() == [False, False, True]
    assert result.tpa_fallback_mask.tolist() == [True, True, True]
    assert result.location_incomplete_mask.tolist() == [True, True, False]


def test_missing_sequence_ids_remain_singletons_inside_location_split():
    from src.models.stp_audit_split import build_location_audit_split

    metadata = [torch.tensor([1, float("nan"), 8]), torch.tensor([1, float("nan"), 8])]
    labels = torch.tensor([0, 1])
    train_counts = torch.tensor([30, 10])

    result = build_location_audit_split(metadata, labels, train_counts, sequence_field_index=1, location_field_index=0)

    assert result.inferential_mask.tolist() == [True, True]
    assert result.tpa_fallback_mask.tolist() == [True, True]
    assert result.sequence_groups == ((0,), (1,))


def test_normalized_loo_mean_excludes_target_and_reuses_singleton_logits():
    from src.models.stp_audit_split import apply_normalized_loo_mean

    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]])
    metadata = _metadata((1,), (1,), (2,))

    actual = apply_normalized_loo_mean(logits, metadata, sequence_field_index=0, eta=0.5)

    assert torch.allclose(actual[0], 0.5 * logits[0] + 0.5 * logits[1])
    assert torch.allclose(actual[1], 0.5 * logits[1] + 0.5 * logits[0])
    assert torch.equal(actual[2], logits[2])


def test_stp_mean_keeps_missing_sequence_ids_as_singletons_and_normalizes_numeric_ids():
    from src.models.stmp_adapter import apply_sequence_consensus

    logits = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, -1.0], [-1.0, 4.0]])
    metadata = [torch.tensor([7]), torch.tensor([7.0]), torch.tensor([float("nan")]), torch.tensor([float("nan")])]

    actual = apply_sequence_consensus(logits, metadata, sequence_field_index=0, eta=1.0)

    assert torch.allclose(actual[0], actual[1])
    assert torch.equal(actual[2], logits[2])
    assert torch.equal(actual[3], logits[3])
