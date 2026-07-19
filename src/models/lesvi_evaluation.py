from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence

import torch

from src.models.loo_bcpd import sequence_groups
from src.models.stmp_adapter import metadata_group_key


BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260718


@dataclass(frozen=True, slots=True)
class MetricSupport:
    class_ids: dict[str, tuple[int, ...]]
    masks: dict[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    low: float
    delta: float
    high: float
    positive_fraction: float
    samples: int
    location_count: int


def fixed_macro_f1(labels: torch.Tensor, predictions: torch.Tensor, class_ids: Sequence[int]) -> float:
    if labels.numel() == 0 or not class_ids:
        return math.nan
    scores: list[torch.Tensor] = []
    for class_id in class_ids:
        truth = labels == int(class_id)
        predicted = predictions == int(class_id)
        numerator = 2.0 * (truth & predicted).sum().float()
        denominator = truth.sum().float() + predicted.sum().float()
        scores.append(torch.where(denominator > 0, numerator / denominator, torch.zeros_like(denominator)))
    return float(torch.stack(scores).mean().item())


def _supported(labels: torch.Tensor, mask: torch.Tensor, *, include_empty: bool) -> tuple[int, ...]:
    values = sorted(int(value) for value in torch.unique(labels[mask]).tolist()) if mask.any() else []
    return tuple(value for value in values if include_empty or value != 0)


def build_metric_support(
    labels: torch.Tensor,
    metadata: Sequence,
    *,
    sequence_field_index: int,
    rotation_common_mask: torch.Tensor | None = None,
) -> MetricSupport:
    labels = labels.long().view(-1)
    if labels.shape[0] != len(metadata):
        raise ValueError("Metric support labels and metadata must align.")
    overall = torch.ones(labels.shape[0], dtype=torch.bool)
    animal = labels != 0
    mixed_empty = torch.zeros_like(overall)
    one_species_no_empty = torch.zeros_like(overall)
    for group in sequence_groups(metadata, sequence_field_index, labels.shape[0]):
        if metadata_group_key(metadata[group[0]], sequence_field_index) is None:
            continue
        indices = torch.tensor(group, dtype=torch.long)
        group_labels = labels[indices]
        nonempty = torch.unique(group_labels[group_labels != 0])
        has_empty = bool((group_labels == 0).any().item())
        if nonempty.numel() == 1 and has_empty:
            mixed_empty[indices] = True
        elif nonempty.numel() == 1 and not has_empty:
            one_species_no_empty[indices] = True
    masks = {
        "overall": overall,
        "animal_true_nonempty": animal,
        "mixed_empty": mixed_empty,
        "one_species_no_empty": one_species_no_empty,
    }
    if rotation_common_mask is not None:
        rotation_common_mask = rotation_common_mask.detach().cpu().bool().view(-1)
        if rotation_common_mask.shape != overall.shape:
            raise ValueError("Rotation common support must align with confirmation labels.")
        masks["rotation_common"] = rotation_common_mask
    class_ids = {
        "overall": _supported(labels, overall, include_empty=True),
        "animal_true_nonempty": _supported(labels, animal, include_empty=False),
        "mixed_empty": _supported(labels, mixed_empty, include_empty=True),
        "one_species_no_empty": _supported(labels, one_species_no_empty, include_empty=False),
    }
    if rotation_common_mask is not None:
        class_ids["rotation_common"] = _supported(labels, rotation_common_mask, include_empty=True)
    return MetricSupport(class_ids=class_ids, masks=masks)


def metric_support_payload(support: MetricSupport, labels: torch.Tensor) -> dict[str, object]:
    return {
        "schema_version": 1,
        "estimands": {
            name: {
                "class_ids": list(support.class_ids[name]),
                "frame_count": int(mask.sum().item()),
                "label_support_count": len(support.class_ids[name]),
                "mask_sha256": hashlib.sha256(mask.detach().cpu().bool().contiguous().numpy().tobytes()).hexdigest(),
            }
            for name, mask in support.masks.items()
        },
        "ground_truth_sha256": hashlib.sha256(labels.detach().cpu().long().contiguous().numpy().tobytes()).hexdigest(),
    }


def write_metric_support(path: Path, support: MetricSupport, labels: torch.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metric_support_payload(support, labels), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_metrics(labels: torch.Tensor, logits: torch.Tensor, support: MetricSupport) -> dict[str, object]:
    predictions = logits.argmax(dim=1)
    metrics: dict[str, object] = {}
    for name, mask in support.masks.items():
        metrics[name] = {
            "macro_f1": fixed_macro_f1(labels[mask], predictions[mask], support.class_ids[name]),
            "top1": math.nan if not mask.any() else float((predictions[mask] == labels[mask]).float().mean().item()),
            "frame_count": int(mask.sum().item()),
        }
    return metrics


def c2w_metrics(labels: torch.Tensor, tpa_logits: torch.Tensor, candidate_logits: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    labels = labels[mask]
    reference = tpa_logits.argmax(dim=1)[mask]
    candidate = candidate_logits.argmax(dim=1)[mask]
    reference_correct = reference == labels
    count = int((reference_correct & (candidate != labels)).sum().item())
    denominator = int(reference_correct.sum().item())
    return {
        "count": count,
        "reference_correct_count": denominator,
        "rate": math.nan if denominator == 0 else count / denominator,
    }


def paired_location_bootstrap(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    location_keys: Sequence[str],
    class_ids: Sequence[int],
    *,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    grouped: dict[str, list[int]] = {}
    for index, key in enumerate(location_keys):
        grouped.setdefault(str(key), []).append(index)
    clusters = tuple(tuple(indices) for _, indices in sorted(grouped.items()))
    if not clusters:
        raise ValueError("LESVI location bootstrap requires at least one location.")
    generator = torch.Generator().manual_seed(seed)
    deltas = torch.empty(samples, dtype=torch.float64)
    for sample_index in range(samples):
        selected = torch.randint(len(clusters), (len(clusters),), generator=generator).tolist()
        rows = torch.tensor([row for cluster_index in selected for row in clusters[cluster_index]], dtype=torch.long)
        reference = fixed_macro_f1(labels[rows], reference_predictions[rows], class_ids)
        candidate = fixed_macro_f1(labels[rows], candidate_predictions[rows], class_ids)
        deltas[sample_index] = candidate - reference
    low, high = torch.quantile(deltas, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return BootstrapResult(
        low=float(low),
        delta=fixed_macro_f1(labels, candidate_predictions, class_ids) - fixed_macro_f1(labels, reference_predictions, class_ids),
        high=float(high),
        positive_fraction=float((deltas > 0).double().mean().item()),
        samples=samples,
        location_count=len(clusters),
    )


def _location_sensitivity(
    labels: torch.Tensor,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    location_keys: Sequence[str],
    class_ids: Sequence[int],
) -> dict[str, object]:
    grouped: dict[str, list[int]] = {}
    for index, key in enumerate(location_keys):
        grouped.setdefault(str(key), []).append(index)
    per_location: dict[str, float] = {}
    leave_one_out: dict[str, float] = {}
    for location, indices in sorted(grouped.items()):
        rows = torch.tensor(indices, dtype=torch.long)
        per_location[location] = fixed_macro_f1(labels[rows], candidate[rows], class_ids) - fixed_macro_f1(labels[rows], reference[rows], class_ids)
        keep = torch.ones(labels.shape[0], dtype=torch.bool)
        keep[rows] = False
        leave_one_out[location] = fixed_macro_f1(labels[keep], candidate[keep], class_ids) - fixed_macro_f1(labels[keep], reference[keep], class_ids)
    values = list(per_location.values())
    return {
        "per_location_delta": per_location,
        "median_delta": math.nan if not values else float(median(values)),
        "positive": sum(value > 0 for value in values),
        "zero": sum(value == 0 for value in values),
        "negative": sum(value < 0 for value in values),
        "leave_one_location_out_delta": leave_one_out,
    }


def build_confirmation_report(
    *,
    labels: torch.Tensor,
    logits_by_name: Mapping[str, torch.Tensor],
    support: MetricSupport,
    location_keys: Sequence[str],
    rotation_logits: Sequence[torch.Tensor],
    rotation_common_mask: torch.Tensor,
    context_eligible_mask: torch.Tensor,
    rotation_fallback_counts: Sequence[int],
) -> dict[str, object]:
    required = {"tpa", "stp_mean", "lesvi", "no_visibility"}
    if not required.issubset(logits_by_name):
        raise ValueError(f"LESVI report is missing required methods: {sorted(required - set(logits_by_name))}.")
    if len(rotation_logits) != 99 or len(rotation_fallback_counts) != 99:
        raise ValueError("LESVI confirmation requires exactly 99 donor rotations.")
    methods = {name: method_metrics(labels, logits, support) for name, logits in logits_by_name.items()}
    overall_ids = support.class_ids["overall"]
    tpa_predictions = logits_by_name["tpa"].argmax(dim=1)
    stp_predictions = logits_by_name["stp_mean"].argmax(dim=1)
    lesvi_predictions = logits_by_name["lesvi"].argmax(dim=1)
    bootstrap = paired_location_bootstrap(labels, stp_predictions, lesvi_predictions, location_keys, overall_ids)
    location_sensitivity = _location_sensitivity(labels, stp_predictions, lesvi_predictions, location_keys, overall_ids)

    mixed_mask = support.masks["mixed_empty"]
    c2w = {
        name: c2w_metrics(labels, logits_by_name["tpa"], logits, mixed_mask)
        for name, logits in logits_by_name.items()
    }
    eligible_count = int(context_eligible_mask.sum().item())
    common_count = int(rotation_common_mask.sum().item())
    rotation_coverage = common_count / max(eligible_count, 1)
    if "rotation_common" not in support.class_ids:
        raise ValueError("Metric support must freeze the rotation_common estimand before predictions are read.")
    rotation_class_ids = support.class_ids["rotation_common"]
    real_delta = fixed_macro_f1(labels[rotation_common_mask], lesvi_predictions[rotation_common_mask], rotation_class_ids) - fixed_macro_f1(labels[rotation_common_mask], tpa_predictions[rotation_common_mask], rotation_class_ids)
    rotation_deltas = [
        fixed_macro_f1(labels[rotation_common_mask], rotated.argmax(dim=1)[rotation_common_mask], rotation_class_ids)
        - fixed_macro_f1(labels[rotation_common_mask], tpa_predictions[rotation_common_mask], rotation_class_ids)
        for rotated in rotation_logits
    ]
    rotation_median = float(median(rotation_deltas))
    p_rotation = (1 + sum(delta >= real_delta for delta in rotation_deltas)) / 100.0

    stp_overall = float(methods["stp_mean"]["overall"]["macro_f1"])
    lesvi_overall = float(methods["lesvi"]["overall"]["macro_f1"])
    no_visibility_overall = float(methods["no_visibility"]["overall"]["macro_f1"])
    mixed_tpa = float(methods["tpa"]["mixed_empty"]["macro_f1"])
    mixed_lesvi = float(methods["lesvi"]["mixed_empty"]["macro_f1"])
    pure_stp = float(methods["stp_mean"]["one_species_no_empty"]["macro_f1"])
    pure_lesvi = float(methods["lesvi"]["one_species_no_empty"]["macro_f1"])
    animal_stp = float(methods["stp_mean"]["animal_true_nonempty"]["macro_f1"])
    animal_lesvi = float(methods["lesvi"]["animal_true_nonempty"]["macro_f1"])
    mixed_rates_finite = all(math.isfinite(float(c2w[name]["rate"])) for name in ("stp_mean", "lesvi", "no_visibility"))
    promotion_checks = {
        "overall_gain_vs_stp_at_least_0p50pp": lesvi_overall - stp_overall >= 0.005,
        "overall_bootstrap_lower_gt_zero": bootstrap.low > 0.0,
        "animal_macro_f1_gt_stp": animal_lesvi > animal_stp,
        "mixed_empty_noninferior_to_tpa_0p25pp": mixed_lesvi >= mixed_tpa - 0.0025,
        "mixed_empty_c2w_lt_stp": mixed_rates_finite and float(c2w["lesvi"]["rate"]) < float(c2w["stp_mean"]["rate"]),
        "one_species_no_empty_noninferior_to_stp_0p50pp": pure_lesvi >= pure_stp - 0.005,
        "rotation_gap_at_least_0p25pp": real_delta - rotation_median >= 0.0025,
        "rotation_p_le_0p05": p_rotation <= 0.05,
        "rotation_coverage_at_least_0p95": rotation_coverage >= 0.95,
        "no_visibility_not_better_overall": no_visibility_overall <= lesvi_overall,
        "no_visibility_mixed_c2w_ge_lesvi": mixed_rates_finite and float(c2w["no_visibility"]["rate"]) >= float(c2w["lesvi"]["rate"]),
    }
    return {
        "method": "LESVI-v0",
        "primary_metric": "fixed-supported-class F1-macro_all",
        "output_roles": {
            "confirmatory": ["tpa", "stp_mean", "lesvi", "no_visibility", "99_donor_event_rotations"],
            "diagnostic_only": [
                "stp_loo_diagnostic",
                "vfep_v0_diagnostic",
                "global_visibility_diagnostic",
                "self_inclusive_diagnostic",
                "location_context_diagnostic",
            ],
            "diagnostics_may_change_promotion": False,
        },
        "methods": methods,
        "mixed_empty_c2w_vs_tpa": c2w,
        "overall_location_bootstrap_lesvi_minus_stp": asdict(bootstrap),
        "location_sensitivity_lesvi_minus_stp": location_sensitivity,
        "rotation": {
            "common_target_count": common_count,
            "context_eligible_count": eligible_count,
            "coverage": rotation_coverage,
            "class_ids": list(rotation_class_ids),
            "real_delta_vs_tpa": real_delta,
            "rotation_deltas_vs_tpa": rotation_deltas,
            "median_rotation_delta_vs_tpa": rotation_median,
            "real_minus_rotation_median": real_delta - rotation_median,
            "p_rotation": p_rotation,
            "fallback_assignment_counts": list(rotation_fallback_counts),
        },
        "promotion": {
            "checks": promotion_checks,
            "passed": all(promotion_checks.values()),
            "id_ood_may_be_opened": all(promotion_checks.values()),
        },
    }


def _equal_count_ece(confidence: torch.Tensor, correct: torch.Tensor, frame_ids: Sequence[str], bins: int) -> float:
    order = sorted(range(confidence.numel()), key=lambda index: (float(confidence[index].item()), str(frame_ids[index])))
    ece = 0.0
    for bin_index in range(bins):
        start = bin_index * len(order) // bins
        end = (bin_index + 1) * len(order) // bins
        if start == end:
            continue
        rows = torch.tensor(order[start:end], dtype=torch.long)
        accuracy = correct[rows].double().mean()
        mean_confidence = confidence[rows].double().mean()
        ece += float((end - start) / len(order) * torch.abs(accuracy - mean_confidence).item())
    return ece


def stable_calibration_report(logits: torch.Tensor, labels: torch.Tensor, frame_ids: Sequence[str], *, bins: int = 15) -> dict[str, float | int]:
    if logits.shape[0] != labels.shape[0] or len(frame_ids) != labels.shape[0] or bins <= 0:
        raise ValueError("Calibration inputs are inconsistent.")
    log_probabilities = torch.log_softmax(logits.double(), dim=1)
    probabilities = log_probabilities.exp()
    confidence, predictions = probabilities.max(dim=1)
    top_label_ece = _equal_count_ece(confidence, predictions == labels, frame_ids, bins)
    nll = float((-log_probabilities[torch.arange(labels.shape[0]), labels]).mean().item())
    one_hot = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).double()
    brier = float(((probabilities - one_hot) ** 2).sum(dim=1).mean().item())
    empty_probability = probabilities[:, 0]
    empty_truth = labels == 0
    binary_confidence = torch.where(empty_probability >= 0.5, empty_probability, 1.0 - empty_probability)
    binary_correct = (empty_probability >= 0.5) == empty_truth
    binary_ece = _equal_count_ece(binary_confidence, binary_correct, frame_ids, bins)
    binary_target = empty_truth.double()
    log_empty = log_probabilities[:, 0]
    log_animal = torch.logsumexp(log_probabilities[:, 1:], dim=1)
    binary_nll = float((-(binary_target * log_empty + (1.0 - binary_target) * log_animal)).mean().item())
    binary_brier = float(((empty_probability - binary_target) ** 2).mean().item())
    return {
        "equal_count_bins": bins,
        "top_label_ece": top_label_ece,
        "multiclass_nll": nll,
        "multiclass_brier": brier,
        "empty_vs_animal_ece": binary_ece,
        "empty_vs_animal_nll": binary_nll,
        "empty_vs_animal_brier": binary_brier,
        "role": "limitation_wording_only",
    }
