"""Burst-aware Tail Evidence Learning primitives."""

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Final, Iterator

import torch
import torch.nn.functional as functional
from torch.utils.data import Sampler


TAIL_MAX_COUNT: Final = 20
MEDIUM_MAX_COUNT: Final = 100
MAX_POSITIVE_CLASS_WEIGHT: Final = 5.0


class BTELConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class BTELArtifacts:
    """Frozen train-split quantities used by the BTEL auxiliary objective."""

    prototypes: torch.Tensor
    class_counts: torch.Tensor
    negative_thresholds: torch.Tensor
    sequence_field_index: int
    empty_class_index: int | None


def sequence_groups(metadata: torch.Tensor, sequence_field_index: int) -> tuple[tuple[int, ...], ...]:
    """Group batch rows by one validated WILDS sequence-metadata column."""
    if metadata.ndim != 2:
        raise BTELConfigurationError("BTEL metadata must have shape [batch, metadata_fields].")
    if sequence_field_index < 0 or sequence_field_index >= metadata.shape[1]:
        raise BTELConfigurationError("BTEL sequence field index is outside metadata columns.")
    groups: dict[int, list[int]] = defaultdict(list)
    for row_index, sequence_id in enumerate(metadata[:, sequence_field_index].detach().cpu().tolist()):
        groups[int(sequence_id)].append(row_index)
    return tuple(tuple(indices) for indices in groups.values())


def class_topk(class_counts: torch.Tensor) -> torch.Tensor:
    """Return one class-specific evidence count for every observed class."""
    counts = class_counts.detach().cpu().long()
    topk = torch.full_like(counts, 3)
    topk[counts <= MEDIUM_MAX_COUNT] = 2
    topk[counts <= TAIL_MAX_COUNT] = 1
    topk[counts <= 0] = 1
    return topk


def leave_one_out_topk_evidence(
    frame_logits: torch.Tensor,
    metadata: torch.Tensor,
    *,
    sequence_field_index: int,
    class_counts: torch.Tensor,
) -> torch.Tensor:
    """Aggregate same-burst class evidence without using a frame's own logit."""
    evidence = torch.zeros_like(frame_logits)
    topk = class_topk(class_counts).to(device=frame_logits.device)
    for group in sequence_groups(metadata, sequence_field_index):
        if len(group) <= 1:
            continue
        group_indices = torch.tensor(group, device=frame_logits.device)
        group_logits = frame_logits.index_select(0, group_indices)
        length = group_logits.shape[0]
        exclude_self = ~torch.eye(length, dtype=torch.bool, device=frame_logits.device)
        masked_logits = group_logits.unsqueeze(0).expand(length, -1, -1).masked_fill(
            ~exclude_self.unsqueeze(2),
            torch.finfo(frame_logits.dtype).min,
        )
        ranked_logits = masked_logits.sort(dim=1, descending=True).values
        ranks = torch.arange(length, device=frame_logits.device).view(1, length, 1)
        per_class_count = topk.clamp_max(length - 1).view(1, 1, -1)
        selected = ranks < per_class_count
        group_evidence = (ranked_logits * selected).sum(dim=1) / per_class_count.squeeze(1)
        evidence[group_indices] = group_evidence.to(dtype=frame_logits.dtype)
    return evidence


def sequence_targets(
    labels: torch.Tensor,
    metadata: torch.Tensor,
    *,
    sequence_field_index: int,
    num_classes: int,
    empty_class_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor, tuple[tuple[int, ...], ...]]:
    """Build multi-label animal targets and a valid-burst mask from frame labels."""
    groups = sequence_groups(metadata, sequence_field_index)
    targets = torch.zeros((len(groups), num_classes), dtype=torch.bool, device=labels.device)
    valid = torch.zeros(len(groups), dtype=torch.bool, device=labels.device)
    for group_index, group in enumerate(groups):
        group_labels = labels[torch.tensor(group, device=labels.device)].unique()
        if empty_class_index is not None:
            group_labels = group_labels[group_labels != empty_class_index]
        if group_labels.numel() == 0:
            continue
        targets[group_index, group_labels.long()] = True
        valid[group_index] = True
    return targets, valid, groups


def frame_targets_from_sequence_targets(
    burst_targets: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    *,
    num_frames: int,
) -> torch.Tensor:
    """Broadcast each burst label set to its frames for framewise calibration."""
    frame_targets = torch.zeros(
        (num_frames, burst_targets.shape[1]),
        dtype=torch.bool,
        device=burst_targets.device,
    )
    for burst_index, group in enumerate(groups):
        frame_targets[torch.tensor(group, device=burst_targets.device)] = burst_targets[burst_index]
    return frame_targets


def negative_evidence_thresholds(
    frame_evidence: torch.Tensor,
    frame_targets: torch.Tensor,
    *,
    quantile: float,
) -> torch.Tensor:
    """Estimate a class-wise evidence floor from frames in bursts without that class."""
    if not 0.0 < quantile < 1.0:
        raise BTELConfigurationError("BTEL negative quantile must be strictly between zero and one.")
    thresholds = torch.empty(frame_evidence.shape[1], dtype=frame_evidence.dtype)
    for class_index in range(frame_evidence.shape[1]):
        negatives = frame_evidence[~frame_targets[:, class_index], class_index]
        thresholds[class_index] = torch.quantile(negatives, quantile) if negatives.numel() else float("inf")
    return thresholds


def negative_thresholds_from_train_evidence(
    frame_logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    *,
    sequence_field_index: int,
    class_counts: torch.Tensor,
    empty_class_index: int | None,
    quantile: float,
) -> torch.Tensor:
    """Calibrate framewise negative floors from leave-one-out train-burst evidence."""
    evidence = leave_one_out_topk_evidence(
        frame_logits,
        metadata,
        sequence_field_index=sequence_field_index,
        class_counts=class_counts,
    )
    burst_targets, _, groups = sequence_targets(
        labels,
        metadata,
        sequence_field_index=sequence_field_index,
        num_classes=frame_logits.shape[1],
        empty_class_index=empty_class_index,
    )
    frame_targets = frame_targets_from_sequence_targets(burst_targets, groups, num_frames=labels.shape[0])
    return negative_evidence_thresholds(evidence, frame_targets, quantile=quantile)


def calibrated_burst_residual(evidence: torch.Tensor, negative_thresholds: torch.Tensor) -> torch.Tensor:
    """Keep only burst evidence exceeding its class-specific negative floor."""
    return (evidence - negative_thresholds.to(device=evidence.device, dtype=evidence.dtype)).clamp_min(0.0)


def positive_class_weights(class_counts: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    """Increase positive loss weight for rare observed classes without changing negatives."""
    counts = class_counts.to(device=device, dtype=torch.float32).clamp_min(1.0)
    weights = (counts.max() / counts).sqrt().clamp_max(MAX_POSITIVE_CLASS_WEIGHT)
    weights[class_counts.to(device=device) <= 0] = 0.0
    return weights


def btel_sequence_loss(
    frame_logits: torch.Tensor,
    labels: torch.Tensor,
    metadata: torch.Tensor,
    *,
    sequence_field_index: int,
    class_counts: torch.Tensor,
    negative_thresholds: torch.Tensor,
    empty_class_index: int | None,
) -> torch.Tensor:
    """Compute class-balanced BCE over calibrated leave-one-out burst evidence."""
    targets, valid, groups = sequence_targets(
        labels,
        metadata,
        sequence_field_index=sequence_field_index,
        num_classes=frame_logits.shape[1],
        empty_class_index=empty_class_index,
    )
    if not valid.any():
        return frame_logits.sum() * 0.0
    evidence = leave_one_out_topk_evidence(
        frame_logits,
        metadata,
        sequence_field_index=sequence_field_index,
        class_counts=class_counts,
    )
    calibrated_logits = frame_logits + calibrated_burst_residual(evidence, negative_thresholds)
    burst_logits = torch.stack(
        [calibrated_logits[torch.tensor(group, device=frame_logits.device)].mean(dim=0) for group in groups]
    )[valid]
    valid_targets = targets[valid].to(dtype=frame_logits.dtype)
    losses = functional.binary_cross_entropy_with_logits(burst_logits, valid_targets, reduction="none")
    weights = torch.ones_like(losses)
    weights = torch.where(valid_targets.bool(), positive_class_weights(class_counts, device=frame_logits.device), weights)
    if empty_class_index is not None:
        weights[:, empty_class_index] = 0.0
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)


class BurstBatchSampler(Sampler[list[int]]):
    """Pack whole metadata-defined bursts under a frame budget.

    Sequences longer than the cap are sampled deterministically for each epoch.
    """

    def __init__(
        self,
        metadata: torch.Tensor,
        *,
        sequence_field_index: int,
        frame_budget: int,
        max_frames_per_sequence: int,
        seed: int,
    ) -> None:
        if frame_budget <= 0 or max_frames_per_sequence <= 0:
            raise BTELConfigurationError("BTEL frame limits must be positive.")
        if max_frames_per_sequence > frame_budget:
            raise BTELConfigurationError("BTEL max frames per sequence cannot exceed the frame budget.")
        self._groups = sequence_groups(metadata, sequence_field_index)
        self._frame_budget = frame_budget
        self._max_frames_per_sequence = max_frames_per_sequence
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _selected_groups(self, epoch: int | None = None) -> Iterator[list[int]]:
        active_epoch = self._epoch if epoch is None else epoch
        rng = random.Random(self._seed + active_epoch)
        groups = list(self._groups)
        rng.shuffle(groups)
        for group in groups:
            selected = list(group)
            rng.shuffle(selected)
            yield selected[: self._max_frames_per_sequence]

    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for selected in self._selected_groups():
            if batch and len(batch) + len(selected) > self._frame_budget:
                yield batch
                batch = []
            batch.extend(selected)
        if batch:
            yield batch

    def batch_count_for_epoch(self, epoch: int) -> int:
        batch_size = 0
        batch_count = 0
        for selected in self._selected_groups(epoch):
            if batch_size and batch_size + len(selected) > self._frame_budget:
                batch_count += 1
                batch_size = 0
            batch_size += len(selected)
        return batch_count + int(batch_size > 0)

    def __len__(self) -> int:
        return self.batch_count_for_epoch(self._epoch)
