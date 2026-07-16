from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Final, Sequence

import torch

from src.models.loo_bcpd import sequence_groups
from src.models.stmp_adapter import metadata_value


AUDIT_SPLIT_SEED: Final[str] = "20260716"
AUDIT_HASH_THRESHOLD: Final[int] = (3 * (1 << 256)) // 5
TAIL_CLASS_MAX_COUNT: Final[int] = 20


class StpAuditSplitError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SplitViability:
    audit_location_count: int
    audit_frame_count: int
    audit_supported_class_fraction: float
    audit_supported_tail_class_count: int
    confirm_location_count: int
    confirm_frame_count: int
    confirm_supported_class_fraction: float
    confirm_supported_tail_class_count: int
    confirm_largest_location_frame_fraction: float
    viability_pass: bool

    def to_public_dict(self) -> dict[str, float | int | bool]:
        return {
            "audit_location_count": self.audit_location_count,
            "audit_frame_count": self.audit_frame_count,
            "audit_supported_class_fraction": self.audit_supported_class_fraction,
            "audit_supported_tail_class_count": self.audit_supported_tail_class_count,
            "confirm_location_count": self.confirm_location_count,
            "confirm_frame_count": self.confirm_frame_count,
            "confirm_supported_class_fraction": self.confirm_supported_class_fraction,
            "confirm_supported_tail_class_count": self.confirm_supported_tail_class_count,
            "confirm_largest_location_frame_fraction": self.confirm_largest_location_frame_fraction,
            "viability_pass": self.viability_pass,
        }


@dataclass(frozen=True, slots=True)
class LocationAuditSplit:
    audit_mask: torch.Tensor
    confirm_mask: torch.Tensor
    inferential_mask: torch.Tensor
    location_incomplete_mask: torch.Tensor
    tpa_fallback_mask: torch.Tensor
    location_keys: tuple[str | None, ...]
    sequence_groups: tuple[tuple[int, ...], ...]
    audit_locations: tuple[str, ...]
    viability: SplitViability


def _metadata_key(metadata_row: torch.Tensor | Sequence[int | float | str] | int | float | str | None, field_index: int | None) -> str | None:
    value = metadata_value(metadata_row, field_index)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value).strip() or None


def _raw_metadata_identity(metadata_row: torch.Tensor | Sequence[int | float | str] | int | float | str | None, field_index: int | None) -> str | None:
    value = metadata_value(metadata_row, field_index)
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    return f"{type(value).__name__}:{value!r}"


def location_split_name(location_key: str) -> str:
    digest = hashlib.sha256(f"{AUDIT_SPLIT_SEED}|{location_key}".encode("utf-8")).hexdigest()
    return "val_audit" if int(digest, 16) < AUDIT_HASH_THRESHOLD else "val_confirm"


def _assert_no_location_collisions(location_keys: Sequence[str | None], raw_locations: Sequence[str | None]) -> None:
    raw_by_key: dict[str, set[str]] = {}
    for location_key, raw_location in zip(location_keys, raw_locations, strict=True):
        if location_key is None or raw_location is None:
            continue
        raw_by_key.setdefault(location_key, set()).add(raw_location)
    collisions = [key for key, values in raw_by_key.items() if len(values) > 1]
    if collisions:
        raise StpAuditSplitError(f"Location canonicalization collision for {collisions[0]!r}.")


def _sequence_location_masks(
    groups: Sequence[Sequence[int]],
    location_keys: Sequence[str | None],
    metadata: Sequence[torch.Tensor],
    sequence_field_index: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    location_incomplete = torch.zeros(len(location_keys), dtype=torch.bool)
    context_eligible = torch.zeros(len(location_keys), dtype=torch.bool)
    for group in groups:
        group_locations = {location_keys[index] for index in group if location_keys[index] is not None}
        has_missing_location = any(location_keys[index] is None for index in group)
        if len(group_locations) > 1:
            raise StpAuditSplitError("A valid sequence spans multiple valid locations.")
        has_valid_sequence = _metadata_key(metadata[group[0]], sequence_field_index) is not None
        index_tensor = torch.tensor(group, dtype=torch.long)
        if has_missing_location:
            location_incomplete[index_tensor] = True
        elif has_valid_sequence and len(group) >= 2:
            context_eligible[index_tensor] = True
    return location_incomplete, context_eligible


def _supported_class_fraction(labels: torch.Tensor, mask: torch.Tensor, full_supported: torch.Tensor) -> float:
    supported = torch.unique(labels[mask]) if mask.any() else torch.empty(0, dtype=labels.dtype)
    return float(supported.numel() / max(int(full_supported.numel()), 1))


def _supported_tail_class_count(labels: torch.Tensor, mask: torch.Tensor, train_class_counts: torch.Tensor) -> int:
    if not mask.any():
        return 0
    supported = torch.unique(labels[mask])
    return int((train_class_counts[supported] <= TAIL_CLASS_MAX_COUNT).sum().item())


def _split_viability(
    labels: torch.Tensor,
    train_class_counts: torch.Tensor,
    audit_mask: torch.Tensor,
    confirm_mask: torch.Tensor,
    location_keys: Sequence[str | None],
) -> SplitViability:
    full_supported = torch.unique(labels)
    audit_locations = {location_keys[index] for index, included in enumerate(audit_mask.tolist()) if included}
    confirm_locations = [location_keys[index] for index, included in enumerate(confirm_mask.tolist()) if included]
    confirm_location_counts: dict[str, int] = {}
    for location_key in confirm_locations:
        if location_key is not None:
            confirm_location_counts[location_key] = confirm_location_counts.get(location_key, 0) + 1
    confirm_frame_count = int(confirm_mask.sum().item())
    largest_confirm_fraction = max(confirm_location_counts.values(), default=0) / max(confirm_frame_count, 1)
    audit_tail = _supported_tail_class_count(labels, audit_mask, train_class_counts)
    confirm_tail = _supported_tail_class_count(labels, confirm_mask, train_class_counts)
    audit_fraction = _supported_class_fraction(labels, audit_mask, full_supported)
    confirm_fraction = _supported_class_fraction(labels, confirm_mask, full_supported)
    viable = (
        len(audit_locations) >= 10
        and len(confirm_location_counts) >= 10
        and audit_tail >= 5
        and confirm_tail >= 5
        and confirm_fraction >= 0.8
        and largest_confirm_fraction <= 0.5
    )
    return SplitViability(
        audit_location_count=len(audit_locations),
        audit_frame_count=int(audit_mask.sum().item()),
        audit_supported_class_fraction=audit_fraction,
        audit_supported_tail_class_count=audit_tail,
        confirm_location_count=len(confirm_location_counts),
        confirm_frame_count=confirm_frame_count,
        confirm_supported_class_fraction=confirm_fraction,
        confirm_supported_tail_class_count=confirm_tail,
        confirm_largest_location_frame_fraction=largest_confirm_fraction,
        viability_pass=viable,
    )


def build_location_audit_split(
    metadata: Sequence[torch.Tensor],
    labels: torch.Tensor,
    train_class_counts: torch.Tensor,
    *,
    sequence_field_index: int | None,
    location_field_index: int | None,
) -> LocationAuditSplit:
    if labels.ndim != 1 or len(metadata) != labels.shape[0]:
        raise StpAuditSplitError("Metadata and labels must describe the same examples.")
    if train_class_counts.ndim != 1:
        raise StpAuditSplitError("Train class counts must be rank-1.")
    location_keys = tuple(_metadata_key(row, location_field_index) for row in metadata)
    raw_locations = tuple(_raw_metadata_identity(row, location_field_index) for row in metadata)
    _assert_no_location_collisions(location_keys, raw_locations)
    groups = sequence_groups(metadata, sequence_field_index, labels.shape[0])
    location_incomplete, context_eligible = _sequence_location_masks(groups, location_keys, metadata, sequence_field_index)
    valid_location = torch.tensor([key is not None for key in location_keys], dtype=torch.bool)
    inferential_mask = valid_location & ~location_incomplete
    audit_mask = torch.zeros_like(inferential_mask)
    confirm_mask = torch.zeros_like(inferential_mask)
    for index, location_key in enumerate(location_keys):
        if not inferential_mask[index] or location_key is None:
            continue
        if location_split_name(location_key) == "val_audit":
            audit_mask[index] = True
        else:
            confirm_mask[index] = True
    viability = _split_viability(labels, train_class_counts, audit_mask, confirm_mask, location_keys)
    audit_locations = tuple(sorted({location_keys[index] for index in torch.where(audit_mask)[0].tolist() if location_keys[index] is not None}))
    return LocationAuditSplit(
        audit_mask=audit_mask,
        confirm_mask=confirm_mask,
        inferential_mask=inferential_mask,
        location_incomplete_mask=location_incomplete,
        tpa_fallback_mask=~context_eligible,
        location_keys=location_keys,
        sequence_groups=groups,
        audit_locations=audit_locations,
        viability=viability,
    )


def apply_normalized_loo_mean(
    logits: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    *,
    sequence_field_index: int | None,
    eta: float,
) -> torch.Tensor:
    if eta == 0.0 or sequence_field_index is None:
        return logits
    output = logits.clone()
    for group in sequence_groups(metadata, sequence_field_index, logits.shape[0]):
        if len(group) <= 1:
            continue
        indices = torch.tensor(group, dtype=torch.long, device=logits.device)
        group_logits = logits.index_select(0, indices)
        loo_mean = (group_logits.sum(dim=0, keepdim=True) - group_logits) / float(len(group) - 1)
        output[indices] = (1.0 - eta) * group_logits + eta * loo_mean
    return output
