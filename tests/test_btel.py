import torch
import pytest


def test_leave_one_out_evidence_excludes_query_frame():
    from src.models.btel import leave_one_out_topk_evidence

    frame_logits = torch.tensor([[10.0, 0.0], [0.0, 8.0], [2.0, 2.0]])
    metadata = torch.tensor([[1], [1], [2]])
    class_counts = torch.tensor([10, 10])

    evidence = leave_one_out_topk_evidence(
        frame_logits,
        metadata,
        sequence_field_index=0,
        class_counts=class_counts,
    )

    assert torch.allclose(evidence[0], torch.tensor([0.0, 8.0]))
    assert torch.allclose(evidence[1], torch.tensor([10.0, 0.0]))
    assert torch.equal(evidence[2], torch.zeros(2))


def test_leave_one_out_evidence_uses_frequency_dependent_topk():
    from src.models.btel import leave_one_out_topk_evidence

    frame_logits = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [3.0, 3.0, 3.0],
            [5.0, 5.0, 5.0],
            [7.0, 7.0, 7.0],
        ]
    )
    metadata = torch.tensor([[1], [1], [1], [1]])
    class_counts = torch.tensor([20, 100, 101])

    evidence = leave_one_out_topk_evidence(
        frame_logits,
        metadata,
        sequence_field_index=0,
        class_counts=class_counts,
    )

    assert torch.equal(evidence[0], torch.tensor([7.0, 6.0, 5.0]))


def test_negative_calibration_suppresses_evidence_below_class_threshold():
    from src.models.btel import calibrated_burst_residual, negative_evidence_thresholds

    sequence_evidence = torch.tensor([[1.0, 4.0], [3.0, 8.0], [5.0, 2.0]])
    sequence_targets = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    thresholds = negative_evidence_thresholds(sequence_evidence, sequence_targets, quantile=0.5)

    residual = calibrated_burst_residual(torch.tensor([[2.0, 3.0]]), thresholds)

    assert torch.allclose(thresholds, torch.tensor([4.0, 3.0]))
    assert torch.equal(residual, torch.zeros(1, 2))


def test_frame_targets_repeat_each_burst_label_set_for_framewise_calibration():
    from src.models.btel import frame_targets_from_sequence_targets

    targets = torch.tensor([[True, False], [False, True]])
    groups = ((0, 1), (2,))

    frame_targets = frame_targets_from_sequence_targets(targets, groups, num_frames=3)

    assert torch.equal(frame_targets, torch.tensor([[True, False], [True, False], [False, True]]))


def test_train_evidence_calibration_uses_framewise_leave_one_out_values():
    from src.models.btel import negative_thresholds_from_train_evidence

    logits = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    labels = torch.tensor([1, 1])
    metadata = torch.tensor([[7], [7]])

    thresholds = negative_thresholds_from_train_evidence(
        logits,
        labels,
        metadata,
        sequence_field_index=0,
        class_counts=torch.tensor([100, 100]),
        empty_class_index=None,
        quantile=0.5,
    )

    assert thresholds[0].item() == 5.0
    assert torch.isinf(thresholds[1])


def test_btel_loss_masks_empty_only_bursts_but_keeps_animal_bursts():
    from src.models.btel import btel_sequence_loss

    frame_logits = torch.tensor([[3.0, 0.0], [2.0, 0.0], [0.0, 4.0]], requires_grad=True)
    labels = torch.tensor([0, 0, 1])
    metadata = torch.tensor([[10], [10], [20]])
    class_counts = torch.tensor([100, 10])
    thresholds = torch.zeros(2)

    loss = btel_sequence_loss(
        frame_logits,
        labels,
        metadata,
        sequence_field_index=0,
        class_counts=class_counts,
        negative_thresholds=thresholds,
        empty_class_index=0,
    )
    loss.backward()

    assert loss.item() > 0.0
    assert torch.equal(frame_logits.grad[:2], torch.zeros_like(frame_logits.grad[:2]))
    assert torch.count_nonzero(frame_logits.grad[2]).item() > 0


def test_burst_batch_sampler_packs_complete_sequences_without_crossing_budget():
    from src.models.btel import BurstBatchSampler

    metadata = torch.tensor([[1], [1], [1], [2], [2], [3]])
    sampler = BurstBatchSampler(
        metadata,
        sequence_field_index=0,
        frame_budget=4,
        max_frames_per_sequence=2,
        seed=7,
    )

    batches = list(sampler)

    flattened = [index for batch in batches for index in batch]
    assert all(len(batch) <= 4 for batch in batches)
    assert len(flattened) == 5
    assert len(set(flattened)) == 5
    assert {3, 4}.issubset(set(flattened))
    assert sum({3, 4}.issubset(set(batch)) for batch in batches) == 1
    assert 5 in flattened
    assert sum(index in {0, 1, 2} for index in flattened) == 2


def test_burst_batch_sampler_length_matches_group_preserving_iteration():
    from src.models.btel import BurstBatchSampler

    sampler = BurstBatchSampler(
        torch.tensor([[1], [1], [2], [2], [3], [3]]),
        sequence_field_index=0,
        frame_budget=3,
        max_frames_per_sequence=2,
        seed=0,
    )

    assert len(sampler) == len(list(sampler)) == 3


def test_planned_training_steps_matches_epoch_specific_burst_batch_counts():
    from src.models.btel import BurstBatchSampler
    from src.train_flyp import planned_training_steps

    metadata = torch.tensor([[0], [1], [2], [3], [4], [4], [4], [4]])
    sampler = BurstBatchSampler(
        metadata,
        sequence_field_index=0,
        frame_budget=5,
        max_frames_per_sequence=5,
        seed=0,
    )
    expected = [sampler.batch_count_for_epoch(epoch) for epoch in (1, 2)]
    args = type("Args", (), {"epochs": 2, "max_train_batches": None})()

    class Loader:
        batch_sampler = sampler

        def __len__(self):
            return len(self.batch_sampler)

    assert len(set(expected)) > 1
    assert planned_training_steps(Loader(), args) == sum(expected)


def test_burst_batch_sampler_rejects_a_sequence_cap_larger_than_frame_budget():
    from src.models.btel import BTELConfigurationError, BurstBatchSampler

    with pytest.raises(BTELConfigurationError, match="cannot exceed"):
        BurstBatchSampler(
            torch.tensor([[1], [1]]),
            sequence_field_index=0,
            frame_budget=1,
            max_frames_per_sequence=2,
            seed=0,
        )


def test_btel_requires_iwildcamval_for_validation_selection():
    from src.models.btel import BTELConfigurationError
    from src.models.btel_artifacts import validate_btel_validation_split

    with pytest.raises(BTELConfigurationError, match="IWildCamVal"):
        validate_btel_validation_split("IWildCamIDVal")


def test_sequence_audit_reports_label_purity_empty_bursts_and_frequency_buckets():
    from src.models.btel_artifacts import audit_sequences

    class Dataset:
        metadata_fields = ["location", "sequence"]

    class Subset:
        dataset = Dataset()
        metadata_array = torch.tensor([[0, 10], [0, 10], [1, 20], [1, 30]])
        y_array = torch.tensor([0, 0, 1, 2])

    audit = audit_sequences(
        Subset(),
        split_name="train",
        num_classes=3,
        classnames=["empty", "deer", "fox"],
    )

    assert audit.sequence_count == 3
    assert audit.singleton_sequences == 2
    assert audit.pure_label_sequences == 3
    assert audit.empty_only_sequences == 1
    assert audit.mixed_empty_sequences == 0
    assert audit.tail_frames == 4
