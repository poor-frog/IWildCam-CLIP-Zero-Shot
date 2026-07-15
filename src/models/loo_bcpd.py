from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Final, Literal, Sequence

import torch
import torch.nn.functional as F

from src.models.stmp_adapter import metadata_value


PRIOR_MASS: Final[float] = 1.0
RESPONSIBILITY_TEMPERATURE: Final[float] = 1.0
_NORM_TOLERANCE: Final[float] = 1e-3


class LooBcpdConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LeaveOneOutSupport:
    weights: torch.Tensor
    feature_sums: torch.Tensor


@dataclass(frozen=True, slots=True)
class ShuffledGroups:
    groups: tuple[tuple[int, ...], ...]
    changed_frame_fraction: float
    unavailable_group_count: int


@dataclass(frozen=True, slots=True)
class LooBcpdResult:
    logits: torch.Tensor
    prototype_scores: torch.Tensor
    valid_burst_count: int
    corrected_frame_count: int
    mean_responsibility_entropy: float
    mean_max_responsibility: float
    mean_effective_class_count: float
    mean_support_mass: float
    mean_rotation_degrees: float
    mean_normalization_penalty: float


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise LooBcpdConfigurationError(f"{name} must be finite.")


def _require_unit_rows(name: str, value: torch.Tensor) -> None:
    _require_finite(name, value)
    norms = value.float().norm(dim=1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=_NORM_TOLERANCE, rtol=0.0):
        raise LooBcpdConfigurationError(f"{name} must contain L2-normalized rows.")


def _metadata_key(metadata_row: torch.Tensor | Sequence[int] | int | float | str | None, field_index: int | None) -> str | None:
    value = metadata_value(metadata_row, field_index)
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def sequence_groups(
    metadata: Sequence[torch.Tensor],
    sequence_field_index: int | None,
    num_examples: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    expected_examples = len(metadata) if num_examples is None else num_examples
    if len(metadata) != expected_examples or sequence_field_index is None:
        return tuple((index,) for index in range(expected_examples))
    grouped: dict[str, list[int]] = defaultdict(list)
    singletons: list[tuple[int, ...]] = []
    for index, metadata_row in enumerate(metadata):
        key = _metadata_key(metadata_row, sequence_field_index)
        if key is None:
            singletons.append((index,))
        else:
            grouped[key].append(index)
    groups = [tuple(indices) for _, indices in sorted(grouped.items(), key=lambda item: item[1][0])]
    return tuple(sorted((*groups, *singletons), key=lambda indices: indices[0]))


def _stable_seed(seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def deterministic_derangement(num_classes: int, seed: int) -> torch.Tensor:
    if num_classes <= 1:
        raise LooBcpdConfigurationError("A class derangement requires at least two classes.")
    generator = torch.Generator().manual_seed(seed)
    offset = int(torch.randint(1, num_classes, (1,), generator=generator).item())
    return (torch.arange(num_classes, dtype=torch.long) + offset) % num_classes


def _time_of_day_key(metadata_row: torch.Tensor, hour_field_index: int | None) -> str:
    hour = metadata_value(metadata_row, hour_field_index)
    if isinstance(hour, (int, float)) and not (isinstance(hour, float) and math.isnan(hour)):
        return "day" if 6 <= int(hour) <= 17 else "night"
    return "unknown"


def shuffled_sequence_groups(
    metadata: Sequence[torch.Tensor],
    sequence_field_index: int | None,
    location_field_index: int | None,
    hour_field_index: int | None,
    seed: int,
) -> ShuffledGroups:
    original_groups = sequence_groups(metadata, sequence_field_index)
    groups_by_location: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    groups_by_stratum: dict[tuple[str, str], list[tuple[int, ...]]] = defaultdict(list)
    for group in original_groups:
        first_row = metadata[group[0]]
        location = _metadata_key(first_row, location_field_index) or "missing-location"
        groups_by_location[location].append(group)
        groups_by_stratum[(location, _time_of_day_key(first_row, hour_field_index))].append(group)

    shuffled_groups: list[tuple[int, ...]] = []
    unavailable = 0
    for location, location_groups in sorted(groups_by_location.items()):
        time_groups = [groups for (candidate_location, _), groups in groups_by_stratum.items() if candidate_location == location]
        candidate_sets = time_groups if all(len(groups) >= 2 for groups in time_groups) else [location_groups]
        for candidate_groups in candidate_sets:
            if len(candidate_groups) == 1:
                shuffled_groups.extend(candidate_groups)
                unavailable += 1
                continue
            source_indices = [index for group in candidate_groups for index in group]
            key = f"shuffle:{location}:{','.join(str(group[0]) for group in candidate_groups)}"
            generator = torch.Generator().manual_seed(_stable_seed(seed, key))
            order = torch.randperm(len(source_indices), generator=generator).tolist()
            shuffled = [source_indices[index] for index in order]
            cursor = 0
            for group in candidate_groups:
                length = len(group)
                shuffled_groups.append(tuple(sorted(shuffled[cursor : cursor + length])))
                cursor += length

    original_membership = {index: frozenset(group) for group in original_groups for index in group}
    shuffled_membership = {index: frozenset(group) for group in shuffled_groups for index in group}
    changed = sum(original_membership[index] != shuffled_membership[index] for index in original_membership)
    return ShuffledGroups(
        groups=tuple(sorted(shuffled_groups, key=lambda group: group[0])),
        changed_frame_fraction=changed / max(len(metadata), 1),
        unavailable_group_count=unavailable,
    )


def leave_one_out_support(
    features: torch.Tensor,
    responsibilities: torch.Tensor,
    group: Sequence[int],
    target_index: int,
    include_target: bool = False,
) -> LeaveOneOutSupport:
    group_indices = torch.tensor(group, dtype=torch.long, device=features.device)
    group_features = features.index_select(0, group_indices).float()
    group_responsibilities = responsibilities.index_select(0, group_indices).float()
    weights = group_responsibilities.sum(dim=0)
    feature_sums = group_responsibilities.t() @ group_features
    if not include_target:
        target_position = group.index(target_index)
        target_feature = group_features[target_position]
        target_responsibilities = group_responsibilities[target_position]
        weights = weights - target_responsibilities
        feature_sums = feature_sums - target_responsibilities.unsqueeze(1) * target_feature.unsqueeze(0)
    return LeaveOneOutSupport(weights=weights, feature_sums=feature_sums)


def _scores_from_support(
    query: torch.Tensor,
    prototypes: torch.Tensor,
    support: LeaveOneOutSupport,
    strength: float,
    variant: Literal["tangent", "unconstrained"],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    denominator = PRIOR_MASS + support.weights
    if torch.any(denominator <= 0.0):
        raise LooBcpdConfigurationError("LOO support denominator must be positive.")
    base_scores = prototypes @ query
    prototype_support_dot = (prototypes * support.feature_sums).sum(dim=1)
    query_support_dot = support.feature_sums @ query
    support_norm_sq = support.feature_sums.square().sum(dim=1)
    match variant:
        case "tangent":
            query_delta = (query_support_dot - base_scores * prototype_support_dot) / denominator
            delta_norm_sq = (support_norm_sq - prototype_support_dot.square()).clamp_min(0.0) / denominator.square()
        case "unconstrained":
            query_delta = (query_support_dot - support.weights * base_scores) / denominator
            delta_norm_sq = (
                support_norm_sq
                - 2.0 * support.weights * prototype_support_dot
                + support.weights.square()
            ).clamp_min(0.0) / denominator.square()
        case unreachable:
            raise LooBcpdConfigurationError(f"Unsupported LOO-BCPD variant: {unreachable!r}")
    normalizer = torch.sqrt(1.0 + strength * strength * delta_norm_sq)
    scores = (base_scores + strength * query_delta) / normalizer
    return scores, query_delta, delta_norm_sq, normalizer


def build_loo_bcpd_logits(
    features: torch.Tensor,
    base_logits: torch.Tensor,
    prototypes: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    *,
    sequence_field_index: int | None,
    prototype_scale: float,
    strength: float,
    cached_tpa_logits: torch.Tensor | None = None,
    variant: Literal["tangent", "unconstrained"] = "tangent",
    include_target: bool = False,
    responsibility_permutation: torch.Tensor | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    normalize_displaced_prototypes: bool = True,
) -> LooBcpdResult:
    if not 0.0 <= float(strength) <= 1.0:
        raise LooBcpdConfigurationError("LOO-BCPD strength must be in [0, 1].")
    if features.ndim != 2 or prototypes.ndim != 2 or base_logits.ndim != 2:
        raise LooBcpdConfigurationError("Features, prototypes, and base logits must be rank-2 tensors.")
    if features.shape[0] != base_logits.shape[0] or features.shape[1] != prototypes.shape[1]:
        raise LooBcpdConfigurationError("LOO-BCPD feature dimensions do not match.")
    if prototypes.shape[0] != base_logits.shape[1]:
        raise LooBcpdConfigurationError("Prototype rows must match base-logit classes.")
    _require_unit_rows("features", features)
    _require_unit_rows("prototypes", prototypes)
    _require_finite("base_logits", base_logits)
    if cached_tpa_logits is not None and cached_tpa_logits.shape != base_logits.shape:
        raise LooBcpdConfigurationError("Cached TPA logits must match base logits.")
    if strength == 0.0 and cached_tpa_logits is not None:
        return LooBcpdResult(
            logits=cached_tpa_logits,
            prototype_scores=features.float() @ prototypes.float().t(),
            valid_burst_count=0,
            corrected_frame_count=0,
            mean_responsibility_entropy=0.0,
            mean_max_responsibility=0.0,
            mean_effective_class_count=0.0,
            mean_support_mass=0.0,
            mean_rotation_degrees=0.0,
            mean_normalization_penalty=0.0,
        )

    normalized_features = features.float()
    normalized_prototypes = prototypes.float()
    responsibilities = F.softmax(base_logits.float() / RESPONSIBILITY_TEMPERATURE, dim=1)
    if responsibility_permutation is not None:
        if responsibility_permutation.shape != (base_logits.shape[1],):
            raise LooBcpdConfigurationError("Responsibility permutation has the wrong shape.")
        responsibilities = responsibilities.index_select(1, responsibility_permutation.to(dtype=torch.long))
    _require_finite("responsibilities", responsibilities)
    base_scores = normalized_features @ normalized_prototypes.t()
    prototype_scores = base_scores.clone()
    logits = base_logits.float() + float(prototype_scale) * prototype_scores
    if cached_tpa_logits is not None:
        logits = cached_tpa_logits.clone().float()
    active_groups = tuple(tuple(group) for group in (groups or sequence_groups(metadata, sequence_field_index, features.shape[0])) if len(group) >= 2)
    if not active_groups:
        return LooBcpdResult(
            logits=logits.to(dtype=base_logits.dtype),
            prototype_scores=prototype_scores.to(dtype=features.dtype),
            valid_burst_count=0,
            corrected_frame_count=0,
            mean_responsibility_entropy=0.0,
            mean_max_responsibility=0.0,
            mean_effective_class_count=0.0,
            mean_support_mass=0.0,
            mean_rotation_degrees=0.0,
            mean_normalization_penalty=0.0,
        )

    entropy = -(responsibilities * responsibilities.clamp_min(1e-12).log()).sum(dim=1)
    entropy_sum = 0.0
    max_probability_sum = 0.0
    effective_class_sum = 0.0
    support_mass_sum = 0.0
    rotation_sum = 0.0
    penalty_sum = 0.0
    corrected_frames = 0
    for group in active_groups:
        for target_index in group:
            support = leave_one_out_support(
                normalized_features,
                responsibilities,
                group,
                target_index,
                include_target=include_target,
            )
            scores, query_delta, delta_norm_sq, normalizer = _scores_from_support(
                normalized_features[target_index],
                normalized_prototypes,
                support,
                float(strength),
                variant,
            )
            if not normalize_displaced_prototypes:
                scores = base_scores[target_index] + float(strength) * query_delta
            prototype_scores[target_index] = scores
            logits[target_index] = base_logits[target_index].float() + float(prototype_scale) * scores
            entropy_sum += float(entropy[target_index].item())
            max_probability_sum += float(responsibilities[target_index].max().item())
            effective_class_sum += float(torch.exp(entropy[target_index]).item())
            support_mass_sum += float(support.weights.mean().item())
            rotation_sum += float(torch.rad2deg(torch.atan(float(strength) * torch.sqrt(delta_norm_sq))).mean().item())
            penalty_sum += float((1.0 - normalizer.reciprocal()).mean().item())
            corrected_frames += 1
    divisor = max(corrected_frames, 1)
    return LooBcpdResult(
        logits=logits.to(dtype=base_logits.dtype),
        prototype_scores=prototype_scores.to(dtype=features.dtype),
        valid_burst_count=len(active_groups),
        corrected_frame_count=corrected_frames,
        mean_responsibility_entropy=entropy_sum / divisor,
        mean_max_responsibility=max_probability_sum / divisor,
        mean_effective_class_count=effective_class_sum / divisor,
        mean_support_mass=support_mass_sum / divisor,
        mean_rotation_degrees=rotation_sum / divisor,
        mean_normalization_penalty=penalty_sum / divisor,
    )


def build_loo_linear_logits(
    features: torch.Tensor,
    base_logits: torch.Tensor,
    prototypes: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    *,
    sequence_field_index: int | None,
    prototype_scale: float,
    strength: float,
    cached_tpa_logits: torch.Tensor | None = None,
    include_target: bool = False,
    responsibility_permutation: torch.Tensor | None = None,
    groups: Sequence[Sequence[int]] | None = None,
) -> LooBcpdResult:
    return build_loo_bcpd_logits(
        features,
        base_logits,
        prototypes,
        metadata,
        sequence_field_index=sequence_field_index,
        prototype_scale=prototype_scale,
        strength=strength,
        cached_tpa_logits=cached_tpa_logits,
        include_target=include_target,
        responsibility_permutation=responsibility_permutation,
        groups=groups,
        normalize_displaced_prototypes=False,
    )
