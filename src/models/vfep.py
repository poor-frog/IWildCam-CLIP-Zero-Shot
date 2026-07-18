from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from src.models.stmp_adapter import INVALID_METADATA_GROUP_KEYS, metadata_group_key


WEIGHT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class VfepResult:
    logits: torch.Tensor
    propagation_strength: torch.Tensor
    support_weight: torch.Tensor
    support_count: torch.Tensor
    target_uncertainty: torch.Tensor
    event_reliability: torch.Tensor
    visibility_capped: torch.Tensor


@dataclass(frozen=True, slots=True)
class VfepShuffle:
    groups: tuple[tuple[int, ...], ...]
    available_frame_mask: tuple[bool, ...]
    changed_frame_fraction: float
    unavailable_group_count: int
    original_pair_retention: float


def _entropy_from_log_probabilities(log_probabilities: torch.Tensor) -> torch.Tensor:
    probabilities = log_probabilities.exp()
    return -(probabilities * log_probabilities).sum(dim=-1)


def _normalized_confidence(log_probabilities: torch.Tensor) -> torch.Tensor:
    class_count = log_probabilities.shape[-1]
    if class_count <= 1:
        return torch.ones(log_probabilities.shape[:-1], device=log_probabilities.device)
    return (1.0 - _entropy_from_log_probabilities(log_probabilities) / math.log(class_count)).clamp(0.0, 1.0)


def _validate_logits(logits: torch.Tensor, empty_class_index: int) -> None:
    if logits.ndim != 2 or logits.shape[1] < 3:
        raise ValueError("VFEP requires [frames, classes] logits with empty plus at least two animal classes.")
    if empty_class_index != 0:
        raise ValueError("VFEP v0 requires the canonical empty class at index 0.")
    if not torch.isfinite(logits).all():
        raise ValueError("VFEP logits must be finite.")


def _sequence_keys(metadata: Sequence, sequence_field_index: int | None) -> tuple[str | None, ...]:
    if sequence_field_index is None or len(metadata) == 0:
        return tuple(None for _ in metadata)
    keys = tuple(metadata_group_key(row, sequence_field_index) for row in metadata)
    return tuple(None if key is None or key.strip().lower() in INVALID_METADATA_GROUP_KEYS else key for key in keys)


def _location_keys(metadata: Sequence, location_field_index: int | None) -> tuple[str | None, ...]:
    if location_field_index is None or len(metadata) == 0:
        return tuple(None for _ in metadata)
    keys = tuple(metadata_group_key(row, location_field_index) for row in metadata)
    return tuple(None if key is None or key.strip().lower() in INVALID_METADATA_GROUP_KEYS else key for key in keys)


def _groups_from_keys(keys: Sequence[str | None]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = {}
    singletons: list[tuple[int, ...]] = []
    for index, key in enumerate(keys):
        if key is None:
            singletons.append((index,))
        else:
            grouped.setdefault(key, []).append(index)
    groups = [tuple(indices) for _, indices in sorted(grouped.items(), key=lambda item: item[1][0])]
    return tuple(sorted((*groups, *singletons), key=lambda group: group[0]))


def _support_indices_by_target(
    metadata: Sequence,
    sequence_field_index: int | None,
    location_field_index: int | None,
    support_scope: Literal["sequence", "location"],
    groups: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    num_examples = len(metadata)
    sequence_keys = _sequence_keys(metadata, sequence_field_index)
    if support_scope == "sequence":
        selected_groups = _groups_from_keys(sequence_keys) if groups is None else tuple(tuple(group) for group in groups)
        supports: list[tuple[int, ...]] = [tuple() for _ in range(num_examples)]
        for group in selected_groups:
            for target in group:
                supports[target] = tuple(group)
        return tuple(supports)

    location_keys = _location_keys(metadata, location_field_index)
    by_location = _groups_from_keys(location_keys)
    supports = [tuple() for _ in range(num_examples)]
    for location_group in by_location:
        for target in location_group:
            target_sequence = sequence_keys[target]
            if target_sequence is None:
                supports[target] = tuple()
                continue
            supports[target] = tuple(
                index for index in location_group
                if sequence_keys[index] != target_sequence
            )
    return tuple(supports)


def build_vfep_logits(
    logits: torch.Tensor,
    metadata: Sequence,
    *,
    sequence_field_index: int | None,
    strength: float,
    empty_class_index: int = 0,
    source_weighting: Literal["reliability", "uniform"] = "reliability",
    aggregation: Literal["log", "arithmetic"] = "log",
    include_target: bool = False,
    factorized: bool = True,
    groups: Sequence[Sequence[int]] | None = None,
    support_scope: Literal["sequence", "location"] = "sequence",
    location_field_index: int | None = None,
) -> VfepResult:
    _validate_logits(logits, empty_class_index)
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("VFEP strength must be in [0, 1].")
    frame_count = logits.shape[0]
    zeros = torch.zeros(frame_count, dtype=torch.float32, device=logits.device)
    false_mask = torch.zeros(frame_count, dtype=torch.bool, device=logits.device)
    if strength == 0.0 or sequence_field_index is None or len(metadata) != frame_count:
        return VfepResult(logits, zeros, zeros, torch.zeros_like(zeros, dtype=torch.long), zeros, zeros, false_mask)

    work_logits = logits.float()
    if factorized:
        log_probabilities = F.log_softmax(work_logits[:, 1:], dim=1)
        full_probabilities = F.softmax(work_logits, dim=1)
        visibility = 1.0 - full_probabilities[:, 0]
    else:
        log_probabilities = F.log_softmax(work_logits, dim=1)
        visibility = torch.ones(frame_count, device=logits.device)
    confidence = _normalized_confidence(log_probabilities)
    weights = visibility * confidence if source_weighting == "reliability" else torch.ones_like(visibility)
    weights = weights.clamp(0.0, 1.0)
    target_uncertainty = (1.0 - confidence).clamp(0.0, 1.0)

    supports = _support_indices_by_target(
        metadata,
        sequence_field_index,
        location_field_index,
        support_scope,
        groups,
    )
    output = logits.clone()
    propagation = zeros.clone()
    support_weight = zeros.clone()
    support_count = torch.zeros(frame_count, dtype=torch.long, device=logits.device)
    event_reliability = zeros.clone()
    visibility_capped = false_mask.clone()

    def reconstruct(target: int, coefficient: torch.Tensor, event_log_probabilities: torch.Tensor) -> torch.Tensor:
        blended = (1.0 - coefficient) * log_probabilities[target] + coefficient * event_log_probabilities
        blended = F.log_softmax(blended, dim=0)
        if factorized:
            partition = torch.logsumexp(work_logits[target, 1:], dim=0)
            candidate = work_logits[target].clone()
            candidate[1:] = partition + blended
            return candidate
        partition = torch.logsumexp(work_logits[target], dim=0)
        return partition + blended

    for target, raw_support in enumerate(supports):
        support = tuple(index for index in raw_support if include_target or index != target)
        if len(support) < 2:
            continue
        index_tensor = torch.tensor(support, dtype=torch.long, device=logits.device)
        selected_weights = weights.index_select(0, index_tensor)
        finite_mask = torch.isfinite(selected_weights) & torch.isfinite(log_probabilities.index_select(0, index_tensor)).all(dim=1)
        if not finite_mask.all():
            index_tensor = index_tensor[finite_mask]
            selected_weights = selected_weights[finite_mask]
        if index_tensor.numel() < 2:
            continue
        total_weight = selected_weights.sum()
        if not torch.isfinite(total_weight) or total_weight <= WEIGHT_EPSILON:
            continue
        selected_log_probabilities = log_probabilities.index_select(0, index_tensor)
        if aggregation == "log":
            event_scores = (selected_weights[:, None] * selected_log_probabilities).sum(dim=0) / total_weight
            event_log_probabilities = F.log_softmax(event_scores, dim=0)
        elif aggregation == "arithmetic":
            event_probabilities = (selected_weights[:, None] * selected_log_probabilities.exp()).sum(dim=0) / total_weight
            event_log_probabilities = torch.log(event_probabilities.clamp_min(WEIGHT_EPSILON))
            event_log_probabilities = F.log_softmax(event_log_probabilities, dim=0)
        else:
            raise ValueError(f"Unsupported VFEP aggregation: {aggregation!r}")
        reliability = _normalized_confidence(event_log_probabilities.unsqueeze(0))[0]
        coefficient = strength * target_uncertainty[target] * reliability
        if coefficient <= 0.0:
            continue
        reconstructed = reconstruct(target, coefficient, event_log_probabilities)
        if factorized:
            original_is_empty = bool(work_logits[target].argmax().item() == empty_class_index)
            candidate_is_empty = bool(reconstructed.argmax().item() == empty_class_index)
            if original_is_empty != candidate_is_empty:
                low = torch.zeros((), device=logits.device)
                high = coefficient
                for _ in range(24):
                    middle = (low + high) / 2.0
                    middle_is_empty = bool(reconstruct(target, middle, event_log_probabilities).argmax().item() == empty_class_index)
                    if middle_is_empty == original_is_empty:
                        low = middle
                    else:
                        high = middle
                coefficient = low
                reconstructed = reconstruct(target, coefficient, event_log_probabilities)
                visibility_capped[target] = True
        output[target] = reconstructed.to(dtype=output.dtype)
        propagation[target] = coefficient
        support_weight[target] = total_weight
        support_count[target] = index_tensor.numel()
        event_reliability[target] = reliability

    return VfepResult(
        output,
        propagation,
        support_weight,
        support_count,
        target_uncertainty,
        event_reliability,
        visibility_capped,
    )


def _stable_generator(seed: int, key: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))


def build_vfep_shuffle_groups(
    metadata: Sequence,
    *,
    sequence_field_index: int,
    location_field_index: int,
    seed: int,
) -> VfepShuffle:
    sequence_keys = _sequence_keys(metadata, sequence_field_index)
    location_keys = _location_keys(metadata, location_field_index)
    original_groups = _groups_from_keys(sequence_keys)
    groups_by_location: dict[str, list[tuple[int, ...]]] = {}
    for group in original_groups:
        location = location_keys[group[0]]
        if location is None or any(location_keys[index] != location for index in group):
            continue
        groups_by_location.setdefault(location, []).append(group)

    shuffled: list[tuple[int, ...]] = []
    available = [False for _ in metadata]
    unavailable = 0
    changed_frames = 0
    retained_pairs = 0
    original_pairs = 0
    for location, location_groups in sorted(groups_by_location.items()):
        if len(location_groups) < 2:
            shuffled.extend((index,) for group in location_groups for index in group)
            unavailable += len(location_groups)
            continue
        lengths = [len(group) for group in location_groups]
        group_order = torch.randperm(len(location_groups), generator=_stable_generator(seed, location)).tolist()
        rotated = group_order[1:] + group_order[:1]
        pools = [list(location_groups[index]) for index in rotated]
        cursors = [0 for _ in pools]
        pseudo_groups: list[tuple[int, ...]] = []
        feasible = True
        for length in lengths:
            members: list[int] = []
            used_sources: set[int] = set()
            while len(members) < length:
                candidates = [idx for idx, pool in enumerate(pools) if cursors[idx] < len(pool) and idx not in used_sources]
                if not candidates:
                    feasible = False
                    break
                source = candidates[0]
                members.append(pools[source][cursors[source]])
                cursors[source] += 1
                used_sources.add(source)
            if not feasible:
                break
            pseudo_groups.append(tuple(sorted(members)))
        if not feasible or sum(map(len, pseudo_groups)) != sum(lengths):
            shuffled.extend((index,) for group in location_groups for index in group)
            unavailable += len(location_groups)
            continue
        shuffled.extend(pseudo_groups)
        for group in pseudo_groups:
            for index in group:
                available[index] = True

    grouped_indices = {index for group in shuffled for index in group}
    for index in range(len(metadata)):
        if index not in grouped_indices:
            shuffled.append((index,))

    original_membership = {index: frozenset(group) for group in original_groups for index in group}
    shuffled_membership = {index: frozenset(group) for group in shuffled for index in group}
    for index, membership in original_membership.items():
        changed_frames += shuffled_membership.get(index, frozenset({index})) != membership
    for group in original_groups:
        for left_position, left in enumerate(group):
            for right in group[left_position + 1 :]:
                original_pairs += 1
                retained_pairs += right in shuffled_membership.get(left, frozenset())
    return VfepShuffle(
        groups=tuple(sorted(shuffled, key=lambda group: group[0])),
        available_frame_mask=tuple(available),
        changed_frame_fraction=changed_frames / max(len(metadata), 1),
        unavailable_group_count=unavailable,
        original_pair_retention=retained_pairs / max(original_pairs, 1),
    )
