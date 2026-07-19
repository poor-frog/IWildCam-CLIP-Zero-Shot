from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
import torch.nn.functional as F

from src.models.loo_bcpd import sequence_groups
from src.models.stmp_adapter import metadata_group_key


METHOD_NAME = "LESVI-v0"
PRIOR_SCHEMA_VERSION = 1
EMPTY_CLASS_INDEX = 0
LOG_EPSILON = 1e-12


class LesviConfigurationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LesviPrior:
    pi: torch.Tensor
    theta: torch.Tensor
    mu: torch.Tensor
    global_visibility: float
    class_mapping_sha256: str
    diagnostics: dict[str, object]


@dataclass(frozen=True, slots=True)
class LesviResult:
    logits: torch.Tensor
    event_posterior: torch.Tensor
    animal_visible_posterior: torch.Tensor
    support_count: torch.Tensor
    metadata_fallback_mask: torch.Tensor
    numerical_fallback_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class DonorRotation:
    seed: int
    support_by_target: tuple[tuple[int, ...], ...]
    available_mask: torch.Tensor
    fallback_assignment_count: int


def _require_prior(prior: LesviPrior, class_count: int) -> None:
    if class_count < 2 or prior.pi.shape != (class_count,) or prior.mu.shape != (class_count,):
        raise LesviConfigurationError("LESVI prior and logits must use the same class count.")
    if prior.theta.shape != (class_count - 1,):
        raise LesviConfigurationError("LESVI theta must contain one value per animal class.")
    for name, value in (("pi", prior.pi), ("theta", prior.theta), ("mu", prior.mu)):
        if not torch.isfinite(value).all():
            raise LesviConfigurationError(f"LESVI {name} must be finite.")
    if (prior.pi <= 0).any() or (prior.mu <= 0).any():
        raise LesviConfigurationError("LESVI pi and mu must be strictly positive.")
    if (prior.theta <= 0).any() or (prior.theta > 1).any():
        raise LesviConfigurationError("LESVI theta must lie in (0, 1].")
    if not math.isfinite(prior.global_visibility) or not 0.0 < prior.global_visibility < 1.0:
        raise LesviConfigurationError("LESVI global visibility must lie strictly between zero and one.")
    if not torch.allclose(prior.pi.double().sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=0.0):
        raise LesviConfigurationError("LESVI pi must sum to one.")
    if not torch.allclose(prior.mu.double().sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-10, rtol=0.0):
        raise LesviConfigurationError("LESVI mu must sum to one.")
    expected_mu = prior_predictive(prior.pi.double(), prior.theta.double())
    if not torch.allclose(prior.mu.double(), expected_mu, atol=1e-10, rtol=0.0):
        raise LesviConfigurationError("LESVI mu does not match pi and theta.")


def prior_predictive(pi: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    pi = pi.double()
    theta = theta.double()
    mu = torch.empty_like(pi)
    mu[0] = pi[0] + (pi[1:] * (1.0 - theta)).sum()
    mu[1:] = pi[1:] * theta
    return mu


def _dataset_identity(labels: torch.Tensor, metadata: Sequence) -> str:
    digest = hashlib.sha256()
    digest.update(labels.detach().cpu().long().contiguous().numpy().tobytes())
    for row in metadata:
        value = row.detach().cpu().tolist() if hasattr(row, "detach") else row.tolist() if hasattr(row, "tolist") else row
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def estimate_lesvi_prior(
    labels: torch.Tensor,
    metadata: Sequence,
    *,
    sequence_field_index: int | None,
    location_field_index: int | None,
    class_count: int,
    class_mapping_sha256: str,
) -> LesviPrior:
    labels = labels.detach().cpu().long().view(-1)
    if labels.shape[0] != len(metadata) or class_count < 2:
        raise LesviConfigurationError("Train labels, metadata, and class count are inconsistent.")
    if labels.numel() == 0 or labels.min() < 0 or labels.max() >= class_count:
        raise LesviConfigurationError("Train labels fall outside the canonical class mapping.")

    animal_frames = torch.zeros(class_count - 1, dtype=torch.float64)
    empty_frames = torch.zeros(class_count - 1, dtype=torch.float64)
    species_event_counts = torch.zeros(class_count - 1, dtype=torch.long)
    per_event_visibility: list[list[float]] = [[] for _ in range(class_count - 1)]
    empty_event_count = 0
    animal_event_count = 0
    excluded_multi_species = 0
    excluded_invalid_sequence = 0
    excluded_invalid_location = 0

    for group in sequence_groups(metadata, sequence_field_index, labels.shape[0]):
        sequence_key = metadata_group_key(metadata[group[0]], sequence_field_index)
        if sequence_key is None:
            excluded_invalid_sequence += 1
            continue
        locations = {metadata_group_key(metadata[index], location_field_index) for index in group}
        if None in locations or len(locations) != 1:
            excluded_invalid_location += 1
            continue
        group_labels = labels[torch.tensor(group, dtype=torch.long)]
        nonempty = torch.unique(group_labels[group_labels != EMPTY_CLASS_INDEX])
        if nonempty.numel() == 0:
            empty_event_count += 1
            continue
        if nonempty.numel() != 1:
            excluded_multi_species += 1
            continue
        class_id = int(nonempty.item())
        if not torch.all((group_labels == EMPTY_CLASS_INDEX) | (group_labels == class_id)):
            excluded_multi_species += 1
            continue
        offset = class_id - 1
        visible = int((group_labels == class_id).sum().item())
        empty = int((group_labels == EMPTY_CLASS_INDEX).sum().item())
        animal_frames[offset] += visible
        empty_frames[offset] += empty
        species_event_counts[offset] += 1
        per_event_visibility[offset].append(visible / max(visible + empty, 1))
        animal_event_count += 1

    total_visible = float(animal_frames.sum().item())
    total_empty = float(empty_frames.sum().item())
    global_visibility = (total_visible + 1.0) / (total_visible + total_empty + 2.0)
    theta = (animal_frames + 1.0) / (animal_frames + empty_frames + 2.0)
    no_support = (animal_frames + empty_frames) == 0
    theta[no_support] = global_visibility

    pi_empty = (empty_event_count + 1.0) / (empty_event_count + animal_event_count + 2.0)
    pi = torch.empty(class_count, dtype=torch.float64)
    pi[0] = pi_empty
    pi[1:] = (1.0 - pi_empty) / float(class_count - 1)
    mu = prior_predictive(pi, theta)

    total_valid_events = empty_event_count + animal_event_count
    empirical_event_frequency = torch.zeros(class_count, dtype=torch.float64)
    empirical_event_frequency[0] = empty_event_count / max(total_valid_events, 1)
    empirical_event_frequency[1:] = species_event_counts.double() / max(total_valid_events, 1)
    uniform_to_empirical = []
    for class_id in range(class_count):
        empirical = float(empirical_event_frequency[class_id].item())
        uniform_to_empirical.append(None if empirical == 0.0 else float(pi[class_id].item() / empirical))
    mean_event_visibility = [
        None if not values else float(sum(values) / len(values))
        for values in per_event_visibility
    ]
    diagnostics: dict[str, object] = {
        "train_dataset_sha256": _dataset_identity(labels, metadata),
        "valid_empty_event_count": empty_event_count,
        "valid_single_species_event_count": animal_event_count,
        "excluded_multi_species_event_count": excluded_multi_species,
        "excluded_invalid_sequence_count": excluded_invalid_sequence,
        "excluded_invalid_location_event_count": excluded_invalid_location,
        "animal_frame_counts": [int(value) for value in animal_frames.tolist()],
        "empty_frame_counts": [int(value) for value in empty_frames.tolist()],
        "species_event_counts": [int(value) for value in species_event_counts.tolist()],
        "pooled_frame_visibility": [float(value) for value in theta.tolist()],
        "mean_per_event_visibility": mean_event_visibility,
        "empirical_event_frequency": [float(value) for value in empirical_event_frequency.tolist()],
        "uniform_to_empirical_prior_ratio": uniform_to_empirical,
        "event_length_weighting": "pooled_frame_counts",
    }
    prior = LesviPrior(pi=pi, theta=theta, mu=mu, global_visibility=global_visibility, class_mapping_sha256=class_mapping_sha256, diagnostics=diagnostics)
    _require_prior(prior, class_count)
    return prior


def prior_to_artifact(prior: LesviPrior) -> dict[str, object]:
    return {
        "method": METHOD_NAME,
        "schema_version": PRIOR_SCHEMA_VERSION,
        "empty_class_index": EMPTY_CLASS_INDEX,
        "class_count": int(prior.pi.numel()),
        "class_mapping_sha256": prior.class_mapping_sha256,
        "pi": [float(value) for value in prior.pi.tolist()],
        "theta": [float(value) for value in prior.theta.tolist()],
        "mu": [float(value) for value in prior.mu.tolist()],
        "global_visibility": float(prior.global_visibility),
        "diagnostics": prior.diagnostics,
    }


def write_prior_artifact(path: Path, prior: LesviPrior) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prior_to_artifact(prior), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_prior_artifact(path: Path, *, expected_class_mapping_sha256: str | None = None) -> LesviPrior:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LesviConfigurationError(f"Invalid LESVI prior artifact: {path}.") from error
    if payload.get("method") != METHOD_NAME or payload.get("schema_version") != PRIOR_SCHEMA_VERSION:
        raise LesviConfigurationError("LESVI prior artifact method or schema is invalid.")
    if payload.get("empty_class_index") != EMPTY_CLASS_INDEX:
        raise LesviConfigurationError("LESVI requires canonical empty class index 0.")
    mapping_checksum = str(payload.get("class_mapping_sha256", ""))
    if expected_class_mapping_sha256 is not None and mapping_checksum != expected_class_mapping_sha256:
        raise LesviConfigurationError("LESVI prior class mapping checksum does not match.")
    class_count = int(payload.get("class_count", 0))
    prior = LesviPrior(
        pi=torch.tensor(payload.get("pi", []), dtype=torch.float64),
        theta=torch.tensor(payload.get("theta", []), dtype=torch.float64),
        mu=torch.tensor(payload.get("mu", []), dtype=torch.float64),
        global_visibility=float(payload.get("global_visibility", math.nan)),
        class_mapping_sha256=mapping_checksum,
        diagnostics=dict(payload.get("diagnostics", {})),
    )
    _require_prior(prior, class_count)
    return prior


def with_visibility_variant(prior: LesviPrior, variant: Literal["species", "global", "none"]) -> LesviPrior:
    if variant == "species":
        return prior
    if variant == "global":
        theta = torch.full_like(prior.theta, prior.global_visibility)
    elif variant == "none":
        theta = torch.ones_like(prior.theta)
    else:
        raise LesviConfigurationError(f"Unsupported visibility variant: {variant!r}.")
    return LesviPrior(
        pi=prior.pi,
        theta=theta,
        mu=prior_predictive(prior.pi, theta),
        global_visibility=prior.global_visibility,
        class_mapping_sha256=prior.class_mapping_sha256,
        diagnostics=prior.diagnostics,
    )


def _log_event_likelihoods(logits: torch.Tensor, prior: LesviPrior) -> torch.Tensor:
    log_b = F.log_softmax(logits.double(), dim=1)
    log_mu = prior.mu.to(device=logits.device, dtype=torch.float64).log()
    theta = prior.theta.to(device=logits.device, dtype=torch.float64)
    log_theta = theta.log()
    log_one_minus = torch.empty_like(theta)
    boundary = theta == 1.0
    log_one_minus[boundary] = -torch.inf
    log_one_minus[~boundary] = torch.log1p(-theta[~boundary])
    empty_component = log_b[:, :1] - log_mu[:1] + log_one_minus.unsqueeze(0)
    animal_component = log_b[:, 1:] - log_mu[1:] + log_theta.unsqueeze(0)
    animal_likelihoods = torch.logaddexp(empty_component, animal_component)
    return torch.cat((log_b[:, :1] - log_mu[:1], animal_likelihoods), dim=1)


def _event_correction(event_scores: torch.Tensor, prior: LesviPrior, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pi = prior.pi.to(device=device, dtype=torch.float64)
    theta = prior.theta.to(device=device, dtype=torch.float64)
    mu = prior.mu.to(device=device, dtype=torch.float64)
    posterior = F.softmax(pi.log() + event_scores, dim=0)
    nu = torch.empty_like(posterior)
    nu[0] = posterior[0] + (posterior[1:] * (1.0 - theta)).sum()
    nu[1:] = posterior[1:] * theta
    return posterior, nu.log() - mu.log()


def build_lesvi_logits(
    logits: torch.Tensor,
    metadata: Sequence,
    *,
    sequence_field_index: int | None,
    prior: LesviPrior,
    include_target: bool = False,
    support_by_target: Sequence[Sequence[int]] | None = None,
) -> LesviResult:
    if logits.ndim != 2 or logits.shape[1] < 2 or len(metadata) != logits.shape[0]:
        raise LesviConfigurationError("LESVI requires matching [frames, classes] logits and metadata.")
    if not torch.isfinite(logits).all():
        raise LesviConfigurationError("Non-finite baseline logits invalidate LESVI confirmation.")
    _require_prior(prior, logits.shape[1])
    frame_count, class_count = logits.shape
    log_likelihoods = _log_event_likelihoods(logits, prior)
    output = logits.clone()
    event_posterior = prior.pi.to(device=logits.device, dtype=torch.float32).repeat(frame_count, 1)
    support_count = torch.zeros(frame_count, dtype=torch.long, device=logits.device)
    metadata_fallback = torch.ones(frame_count, dtype=torch.bool, device=logits.device)
    numerical_fallback = torch.zeros(frame_count, dtype=torch.bool, device=logits.device)

    if support_by_target is not None:
        if len(support_by_target) != frame_count:
            raise LesviConfigurationError("Custom LESVI supports must contain one entry per target.")
        supports = tuple(tuple(int(index) for index in support) for support in support_by_target)
        for target, support in enumerate(supports):
            selected = tuple(index for index in support if include_target or index != target)
            if not selected:
                continue
            if min(selected) < 0 or max(selected) >= frame_count:
                raise LesviConfigurationError("Custom LESVI support index is out of range.")
            indices = torch.tensor(selected, dtype=torch.long, device=logits.device)
            scores = log_likelihoods.index_select(0, indices).sum(dim=0)
            metadata_fallback[target] = False
            posterior, correction = _event_correction(scores, prior, logits.device)
            candidate = logits[target].float() + correction.float()
            if not torch.isfinite(candidate).all():
                numerical_fallback[target] = True
                continue
            output[target] = candidate.to(dtype=output.dtype)
            event_posterior[target] = posterior.float()
            support_count[target] = len(selected)
            metadata_fallback[target] = False
    else:
        if sequence_field_index is not None:
            for group in sequence_groups(metadata, sequence_field_index, frame_count):
                sequence_key = metadata_group_key(metadata[group[0]], sequence_field_index)
                if sequence_key is None or len(group) <= 1 and not include_target:
                    continue
                indices = torch.tensor(group, dtype=torch.long, device=logits.device)
                group_likelihoods = log_likelihoods.index_select(0, indices)
                total = group_likelihoods.sum(dim=0)
                for position, target in enumerate(group):
                    scores = total if include_target else total - group_likelihoods[position]
                    count = len(group) if include_target else len(group) - 1
                    if count <= 0:
                        continue
                    metadata_fallback[target] = False
                    posterior, correction = _event_correction(scores, prior, logits.device)
                    candidate = logits[target].float() + correction.float()
                    if not torch.isfinite(candidate).all():
                        numerical_fallback[target] = True
                        continue
                    output[target] = candidate.to(dtype=output.dtype)
                    event_posterior[target] = posterior.float()
                    support_count[target] = count
                    metadata_fallback[target] = False

    animal_visible = 1.0 - F.softmax(output.float(), dim=1)[:, EMPTY_CLASS_INDEX]
    return LesviResult(
        logits=output,
        event_posterior=event_posterior,
        animal_visible_posterior=animal_visible,
        support_count=support_count,
        metadata_fallback_mask=metadata_fallback,
        numerical_fallback_mask=numerical_fallback,
    )


def build_location_context_supports(
    metadata: Sequence,
    *,
    sequence_field_index: int,
    location_field_index: int,
) -> tuple[tuple[int, ...], ...]:
    sequence_keys = tuple(metadata_group_key(row, sequence_field_index) for row in metadata)
    location_keys = tuple(metadata_group_key(row, location_field_index) for row in metadata)
    supports: list[tuple[int, ...]] = [tuple() for _ in metadata]
    by_location: dict[str, list[int]] = {}
    for index, location in enumerate(location_keys):
        if location is not None:
            by_location.setdefault(location, []).append(index)
    for target, (sequence, location) in enumerate(zip(sequence_keys, location_keys, strict=True)):
        if sequence is None or location is None:
            continue
        supports[target] = tuple(index for index in by_location[location] if sequence_keys[index] != sequence)
    return tuple(supports)


def _stable_generator(seed: int, key: str) -> torch.Generator:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).digest()
    return torch.Generator().manual_seed(int.from_bytes(digest[:8], byteorder="big", signed=False))


def _length_bin(length: int) -> int:
    if length == 2:
        return 0
    if length <= 5:
        return 1
    if length <= 10:
        return 2
    return 3


def _sattolo(items: Sequence[int], generator: torch.Generator) -> list[int]:
    shuffled = list(items)
    for index in range(len(shuffled) - 1, 0, -1):
        replacement = int(torch.randint(index, (1,), generator=generator).item())
        shuffled[index], shuffled[replacement] = shuffled[replacement], shuffled[index]
    return shuffled


def build_donor_event_rotation(
    metadata: Sequence,
    *,
    sequence_field_index: int,
    location_field_index: int,
    seed: int,
) -> DonorRotation:
    groups = sequence_groups(metadata, sequence_field_index, len(metadata))
    eligible = [
        group for group in groups
        if len(group) >= 2
        and metadata_group_key(metadata[group[0]], sequence_field_index) is not None
        and metadata_group_key(metadata[group[0]], location_field_index) is not None
    ]
    by_location: dict[str, list[tuple[int, ...]]] = {}
    for group in eligible:
        location = metadata_group_key(metadata[group[0]], location_field_index)
        assert location is not None
        if any(metadata_group_key(metadata[index], location_field_index) != location for index in group):
            raise LesviConfigurationError("A LESVI event spans multiple locations.")
        by_location.setdefault(location, []).append(group)

    support_by_target: list[tuple[int, ...]] = [tuple() for _ in metadata]
    available = torch.zeros(len(metadata), dtype=torch.bool)
    fallback_assignments = 0
    for location, location_groups in sorted(by_location.items()):
        if len(location_groups) < 2:
            continue
        bins: dict[int, list[tuple[int, ...]]] = {}
        for group in location_groups:
            bins.setdefault(_length_bin(len(group)), []).append(group)
        strata: list[list[tuple[int, ...]]] = []
        leftovers: list[tuple[int, ...]] = []
        for _, bin_groups in sorted(bins.items()):
            if len(bin_groups) >= 2:
                strata.append(bin_groups)
            else:
                leftovers.extend(bin_groups)
        if leftovers:
            if len(leftovers) >= 2:
                strata.append(leftovers)
                fallback_assignments += len(leftovers)
            else:
                nearest = min(strata, key=lambda values: abs(len(values[0]) - len(leftovers[0]))) if strata else list(location_groups)
                if nearest in strata:
                    strata.remove(nearest)
                strata.append([*nearest, *leftovers])
                fallback_assignments += len(nearest) + len(leftovers)
        for stratum_index, stratum in enumerate(strata):
            if len(stratum) < 2:
                continue
            ordered = sorted(stratum, key=lambda group: (len(group), group[0]))
            donors = _sattolo(list(range(len(ordered))), _stable_generator(seed, f"{location}:{stratum_index}"))
            for target_position, donor_position in enumerate(donors):
                target_group = ordered[target_position]
                donor_group = ordered[donor_position]
                if target_group == donor_group:
                    raise LesviConfigurationError("LESVI donor rotation produced a fixed point.")
                for target in target_group:
                    support_by_target[target] = tuple(donor_group)
                    available[target] = True
    return DonorRotation(
        seed=seed,
        support_by_target=tuple(support_by_target),
        available_mask=available,
        fallback_assignment_count=fallback_assignments,
    )


def verify_lesvi_synthetic_reference() -> dict[str, float | bool]:
    pi = torch.tensor([0.3, 0.35, 0.35], dtype=torch.float64)
    theta = torch.tensor([0.6, 0.8], dtype=torch.float64)
    prior = LesviPrior(
        pi=pi,
        theta=theta,
        mu=prior_predictive(pi, theta),
        global_visibility=0.7,
        class_mapping_sha256="synthetic",
        diagnostics={},
    )
    logits = torch.tensor([[0.2, 1.1, -0.7], [1.0, 0.4, -0.2], [-0.4, 0.3, 1.2]], dtype=torch.float32)
    metadata = [torch.tensor([1, 5]) for _ in range(3)]
    result = build_lesvi_logits(logits, metadata, sequence_field_index=1, prior=prior)
    likelihood_error = float(((_log_event_likelihoods(logits, prior).exp() @ pi) - 1.0).abs().max().item())

    b = F.softmax(logits.double(), dim=1)
    psi = torch.zeros(3, 3, dtype=torch.float64)
    psi[0, 0] = 1.0
    psi[1, 0], psi[1, 1] = 1.0 - theta[0], theta[0]
    psi[2, 0], psi[2, 2] = 1.0 - theta[1], theta[1]
    reference: list[torch.Tensor] = []
    for target in range(3):
        event = pi.clone()
        for context in range(3):
            if context != target:
                event *= (psi * (b[context] / prior.mu).unsqueeze(0)).sum(dim=1)
        event /= event.sum()
        posterior = b[target] * (event @ psi) / prior.mu
        reference.append(posterior / posterior.sum())
    posterior_error = float(
        (F.softmax(result.logits.double(), dim=1) - torch.stack(reference)).abs().max().item()
    )

    changed = logits.clone()
    changed[0] = torch.tensor([9.0, -7.0, 2.0])
    changed_result = build_lesvi_logits(changed, metadata, sequence_field_index=1, prior=prior)
    loo_error = float((result.event_posterior[0] - changed_result.event_posterior[0]).abs().max().item())
    no_visibility = build_lesvi_logits(
        logits,
        metadata,
        sequence_field_index=1,
        prior=with_visibility_variant(prior, "none"),
    )
    passed = likelihood_error <= 1e-10 and posterior_error <= 1e-6 and loo_error <= 1e-7 and bool(torch.isfinite(no_visibility.logits).all())
    if not passed:
        raise LesviConfigurationError("LESVI synthetic verification failed; frozen specification was not created.")
    return {
        "passed": passed,
        "likelihood_normalization_max_error": likelihood_error,
        "pseudo_joint_posterior_max_error": posterior_error,
        "loo_target_mutation_max_error": loo_error,
        "no_visibility_finite": True,
    }
