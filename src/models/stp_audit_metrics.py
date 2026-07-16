from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import torch


LOGIT_CHANGE_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class BootstrapSummary:
    low: float
    delta: float
    high: float
    positive_delta_fraction: float
    requested_samples: int
    valid_samples: int
    location_count: int
    median_class_coverage: float
    minimum_class_coverage: float
    inferential: bool


def macro_f1(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> float:
    if labels.numel() == 0:
        return 0.0
    encoded = labels * num_classes + predictions
    confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes).float()
    f1 = 2.0 * confusion.diag() / (confusion.sum(dim=0) + confusion.sum(dim=1)).clamp_min(1.0)
    active = confusion.sum(dim=1) > 0
    return f1[active].mean().item() if active.any() else 0.0


def _clusters(cluster_keys: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    grouped: dict[str, list[int]] = {}
    for index, cluster_key in enumerate(cluster_keys):
        grouped.setdefault(cluster_key, []).append(index)
    return tuple(tuple(indices) for _, indices in sorted(grouped.items()))


def _paired_cluster_bootstrap(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    cluster_keys: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
    cluster_label: str,
) -> BootstrapSummary:
    if labels.ndim != 1 or labels.shape != reference_predictions.shape or labels.shape != candidate_predictions.shape:
        raise ValueError("Bootstrap labels and predictions must describe the same frames.")
    if len(cluster_keys) != labels.shape[0] or bootstrap_samples <= 0:
        raise ValueError(f"Bootstrap {cluster_label} keys and sample count are invalid.")
    clusters = _clusters(cluster_keys)
    if not clusters:
        raise ValueError(f"{cluster_label.title()} bootstrap requires at least one cluster.")
    num_classes = int(torch.maximum(labels.max(), torch.maximum(reference_predictions.max(), candidate_predictions.max())).item()) + 1
    original_supported = max(int(labels.unique().numel()), 1)
    generator = torch.Generator().manual_seed(seed)
    deltas: list[float] = []
    coverages: list[float] = []
    for _ in range(bootstrap_samples):
        selected_clusters = torch.randint(len(clusters), (len(clusters),), generator=generator).tolist()
        sampled_rows = torch.tensor([row for cluster_index in selected_clusters for row in clusters[cluster_index]], dtype=torch.long)
        sampled_labels = labels[sampled_rows]
        if sampled_labels.unique().numel() < 2:
            continue
        reference_f1 = macro_f1(sampled_labels, reference_predictions[sampled_rows], num_classes)
        candidate_f1 = macro_f1(sampled_labels, candidate_predictions[sampled_rows], num_classes)
        if not torch.isfinite(torch.tensor((reference_f1, candidate_f1))).all():
            continue
        deltas.append(candidate_f1 - reference_f1)
        coverages.append(float(sampled_labels.unique().numel() / original_supported))
    if not deltas:
        raise ValueError("Location bootstrap produced no valid replicates.")
    delta_tensor = torch.tensor(deltas)
    coverage_tensor = torch.tensor(coverages)
    quantiles = torch.quantile(delta_tensor, torch.tensor([0.025, 0.975]))
    return BootstrapSummary(
        low=quantiles[0].item(),
        delta=macro_f1(labels, candidate_predictions, num_classes) - macro_f1(labels, reference_predictions, num_classes),
        high=quantiles[1].item(),
        positive_delta_fraction=(delta_tensor > 0.0).float().mean().item(),
        requested_samples=bootstrap_samples,
        valid_samples=len(deltas),
        location_count=len(clusters),
        median_class_coverage=coverage_tensor.median().item(),
        minimum_class_coverage=coverage_tensor.min().item(),
        inferential=len(clusters) >= 10 and len(deltas) >= min(1000, bootstrap_samples),
    )


def paired_location_bootstrap(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    location_keys: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> BootstrapSummary:
    return _paired_cluster_bootstrap(
        labels,
        reference_predictions,
        candidate_predictions,
        location_keys,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        cluster_label="location",
    )


def paired_sequence_bootstrap(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    sequence_keys: Sequence[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> BootstrapSummary:
    return _paired_cluster_bootstrap(
        labels,
        reference_predictions,
        candidate_predictions,
        sequence_keys,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        cluster_label="sequence",
    )


def _signal_values(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(logits, dim=1)
    confidence, _ = probabilities.max(dim=1)
    top_two = probabilities.topk(k=2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return probabilities, confidence, margin, entropy


def _frequency_bin_name(count: int) -> str:
    if count <= 20:
        return "tail"
    if count <= 100:
        return "medium"
    return "head"


def transition_summary(
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    train_class_counts: torch.Tensor,
) -> dict[str, object]:
    reference = reference_logits.argmax(dim=1)
    candidate = candidate_logits.argmax(dim=1)
    changed = reference != candidate
    wrong_to_correct = (reference != labels) & (candidate == labels)
    correct_to_wrong = (reference == labels) & (candidate != labels)
    predicted_tail = train_class_counts[reference] <= 20
    candidate_head = train_class_counts[candidate] >= 101
    frequency_movements: dict[str, int] = {}
    for reference_count, candidate_count in zip(train_class_counts[reference].tolist(), train_class_counts[candidate].tolist(), strict=True):
        transition = f"{_frequency_bin_name(int(reference_count))}_to_{_frequency_bin_name(int(candidate_count))}"
        frequency_movements[transition] = frequency_movements.get(transition, 0) + 1
    return {
        "changed_prediction_count": int(changed.sum().item()),
        "unchanged_prediction_count": int((~changed).sum().item()),
        "changed_prediction_fraction": float(changed.float().mean().item()),
        "wrong_to_correct_count": int(wrong_to_correct.sum().item()),
        "correct_to_wrong_count": int(correct_to_wrong.sum().item()),
        "empty_to_animal_count": int(((reference == 0) & (candidate != 0)).sum().item()),
        "animal_to_empty_count": int(((reference != 0) & (candidate == 0)).sum().item()),
        "species_to_species_count": int(((reference != 0) & (candidate != 0) & changed).sum().item()),
        "predicted_tail_to_head_count": int((predicted_tail & candidate_head & changed).sum().item()),
        "predicted_frequency_bin_movements": frequency_movements,
        "effective_logit_change_count": int((reference_logits.sub(candidate_logits).abs().amax(dim=1) > LOGIT_CHANGE_EPSILON).sum().item()),
    }


def comparison_payload(
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    location_keys: Sequence[str],
    train_class_counts: torch.Tensor,
    *,
    bootstrap_samples: int,
    seed: int,
    sequence_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    reference_predictions = reference_logits.argmax(dim=1)
    candidate_predictions = candidate_logits.argmax(dim=1)
    _, confidence, margin, entropy = _signal_values(reference_logits)
    changed = reference_predictions != candidate_predictions
    wrong_to_correct = (reference_predictions != labels) & (candidate_predictions == labels)
    correct_to_wrong = (reference_predictions == labels) & (candidate_predictions != labels)
    bootstrap = paired_location_bootstrap(
        labels,
        reference_predictions,
        candidate_predictions,
        location_keys,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    payload = {
        "reference_macro_f1": macro_f1(labels, reference_predictions, reference_logits.shape[1]),
        "candidate_macro_f1": macro_f1(labels, candidate_predictions, candidate_logits.shape[1]),
        "location_bootstrap": asdict(bootstrap),
        "transitions": transition_summary(labels, reference_logits, candidate_logits, train_class_counts),
        "signals": {
            "wrong_to_correct": _signal_summary(confidence, margin, entropy, wrong_to_correct),
            "correct_to_wrong": _signal_summary(confidence, margin, entropy, correct_to_wrong),
            "changed": _signal_summary(confidence, margin, entropy, changed),
        },
    }
    if sequence_keys is not None:
        sequence_bootstrap = paired_sequence_bootstrap(
            labels,
            reference_predictions,
            candidate_predictions,
            sequence_keys,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        payload["sequence_bootstrap_sensitivity"] = asdict(sequence_bootstrap)
    return payload


def _signal_summary(confidence: torch.Tensor, margin: torch.Tensor, entropy: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    if not mask.any():
        return {"count": 0, "confidence_median": 0.0, "margin_median": 0.0, "entropy_median": 0.0}
    return {
        "count": int(mask.sum().item()),
        "confidence_median": confidence[mask].median().item(),
        "margin_median": margin[mask].median().item(),
        "entropy_median": entropy[mask].median().item(),
    }


def build_audit_payload(
    *,
    split_name: str,
    comparison_payloads: Mapping[str, Mapping[str, object]],
    viability_payload: Mapping[str, float | int | bool],
    audit_bin_edges: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    if split_name != "val_audit":
        raise ValueError("Phase A may only materialize val_audit performance payloads.")
    return {
        "split": split_name,
        "comparisons": {name: dict(payload) for name, payload in comparison_payloads.items()},
        "viability": dict(viability_payload),
        "audit_bin_edges": {name: list(values) for name, values in audit_bin_edges.items()},
    }
