from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Mapping, Sequence

import torch

from src.models.stmp_adapter import apply_sequence_consensus
from src.models.stp_audit_metrics import paired_location_bootstrap
from src.models.vfep import VfepResult, build_vfep_logits, build_vfep_shuffle_groups


TIE_TOLERANCE = 0.0001


@dataclass(frozen=True, slots=True)
class FixedBootstrap:
    low: float
    delta: float
    high: float
    positive_fraction: float
    samples: int
    location_count: int


def fixed_class_macro_f1(labels: torch.Tensor, predictions: torch.Tensor, class_ids: Sequence[int]) -> float:
    scores: list[torch.Tensor] = []
    for class_id in class_ids:
        truth = labels == class_id
        predicted = predictions == class_id
        true_positive = (truth & predicted).sum().float()
        denominator = truth.sum().float() + predicted.sum().float()
        scores.append(torch.where(denominator > 0, 2.0 * true_positive / denominator, torch.zeros_like(denominator)))
    return float(torch.stack(scores).mean().item()) if scores else 0.0


def paired_fixed_class_location_bootstrap(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    location_keys: Sequence[str],
    class_ids: Sequence[int],
    *,
    samples: int = 2000,
    seed: int = 20260718,
) -> FixedBootstrap:
    grouped: dict[str, list[int]] = {}
    for index, key in enumerate(location_keys):
        grouped.setdefault(str(key), []).append(index)
    clusters = tuple(tuple(indices) for _, indices in sorted(grouped.items()))
    if not clusters:
        raise ValueError("VFEP location bootstrap requires at least one location.")
    generator = torch.Generator().manual_seed(seed)
    deltas = torch.empty(samples, dtype=torch.float32)
    for sample_index in range(samples):
        selected = torch.randint(len(clusters), (len(clusters),), generator=generator).tolist()
        rows = torch.tensor([row for cluster_index in selected for row in clusters[cluster_index]], dtype=torch.long)
        reference = fixed_class_macro_f1(labels[rows], reference_predictions[rows], class_ids)
        candidate = fixed_class_macro_f1(labels[rows], candidate_predictions[rows], class_ids)
        deltas[sample_index] = candidate - reference
    low, high = torch.quantile(deltas, torch.tensor([0.025, 0.975])).tolist()
    return FixedBootstrap(
        low=float(low),
        delta=fixed_class_macro_f1(labels, candidate_predictions, class_ids)
        - fixed_class_macro_f1(labels, reference_predictions, class_ids),
        high=float(high),
        positive_fraction=float((deltas > 0).float().mean().item()),
        samples=samples,
        location_count=len(clusters),
    )


def _metrics(labels: torch.Tensor, logits: torch.Tensor, class_ids: Sequence[int], empty_index: int = 0) -> dict:
    predictions = logits.argmax(dim=1)
    animal_classes = tuple(class_id for class_id in class_ids if class_id != empty_index)
    animal_mask = labels != empty_index
    return {
        "macro_f1": fixed_class_macro_f1(labels, predictions, class_ids),
        "top1": float((predictions == labels).float().mean().item()),
        "animal_macro_f1": fixed_class_macro_f1(labels[animal_mask], predictions[animal_mask], animal_classes),
    }


def _select_strength(rows: list[dict]) -> dict:
    best = rows[0]
    for row in rows[1:]:
        if row["macro_f1"] > best["macro_f1"] + TIE_TOLERANCE:
            best = row
        elif abs(row["macro_f1"] - best["macro_f1"]) <= TIE_TOLERANCE and row["strength"] < best["strength"]:
            best = row
    return best


def _species_transitions(labels: torch.Tensor, reference: torch.Tensor, candidate: torch.Tensor, empty_index: int) -> dict:
    animal = labels != empty_index
    both_animal = animal & (reference != empty_index) & (candidate != empty_index)
    wrong_to_correct = both_animal & (reference != labels) & (candidate == labels)
    correct_to_wrong = both_animal & (reference == labels) & (candidate != labels)
    return {
        "wrong_to_correct": int(wrong_to_correct.sum().item()),
        "correct_to_wrong": int(correct_to_wrong.sum().item()),
        "net": int(wrong_to_correct.sum().item() - correct_to_wrong.sum().item()),
    }


def _empty_confusion(labels: torch.Tensor, predictions: torch.Tensor, empty_index: int) -> dict:
    truth = labels == empty_index
    predicted = predictions == empty_index
    true_positive = int((truth & predicted).sum().item())
    false_positive = int((~truth & predicted).sum().item())
    false_negative = int((truth & ~predicted).sum().item())
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "f1": 0.0 if denominator == 0 else 2 * true_positive / denominator,
    }


def _visibility_invariants(labels: torch.Tensor, tpa_logits: torch.Tensor, vfep_logits: torch.Tensor, empty_index: int) -> dict:
    tpa_probabilities = torch.softmax(tpa_logits.float(), dim=1)
    vfep_probabilities = torch.softmax(vfep_logits.float(), dim=1)
    tpa_predictions = tpa_logits.argmax(dim=1)
    vfep_predictions = vfep_logits.argmax(dim=1)
    empty_state_equal = (tpa_predictions == empty_index) == (vfep_predictions == empty_index)
    tpa_confusion = _empty_confusion(labels, tpa_predictions, empty_index)
    vfep_confusion = _empty_confusion(labels, vfep_predictions, empty_index)
    return {
        "max_empty_posterior_error": float((tpa_probabilities[:, empty_index] - vfep_probabilities[:, empty_index]).abs().max().item()),
        "empty_state_changes": int((~empty_state_equal).sum().item()),
        "empty_predictions_equal": bool(empty_state_equal.all().item()),
        "empty_to_animal": int(((tpa_predictions == empty_index) & (vfep_predictions != empty_index)).sum().item()),
        "animal_to_empty": int(((tpa_predictions != empty_index) & (vfep_predictions == empty_index)).sum().item()),
        "tpa_empty_confusion": tpa_confusion,
        "vfep_empty_confusion": vfep_confusion,
        "empty_confusion_equal": tpa_confusion == vfep_confusion,
    }


def _equal_location_top1(
    labels: torch.Tensor,
    reference_predictions: torch.Tensor,
    candidate_predictions: torch.Tensor,
    location_keys: Sequence[str],
) -> dict:
    grouped: dict[str, list[int]] = {}
    for index, key in enumerate(location_keys):
        grouped.setdefault(str(key), []).append(index)
    deltas = []
    for _, indices in sorted(grouped.items()):
        rows = torch.tensor(indices, dtype=torch.long)
        reference = (reference_predictions[rows] == labels[rows]).float().mean()
        candidate = (candidate_predictions[rows] == labels[rows]).float().mean()
        deltas.append(float((candidate - reference).item()))
    ordered = sorted(deltas)
    return {
        "location_count": len(ordered),
        "median_delta": float(median(ordered)) if ordered else 0.0,
        "positive": sum(value > 0.0 for value in ordered),
        "zero": sum(value == 0.0 for value in ordered),
        "negative": sum(value < 0.0 for value in ordered),
    }


def build_vfep_pilot_report(
    *,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    metadata: Sequence,
    location_keys: Sequence[str],
    sequence_field_index: int,
    location_field_index: int,
    strengths: Sequence[float],
    stp_strengths: Sequence[float],
    bootstrap_samples: int,
    bootstrap_seed: int,
    shuffle_seeds: Sequence[int],
    empty_index: int = 0,
) -> tuple[dict, Mapping[str, torch.Tensor]]:
    class_ids = tuple(sorted(int(value) for value in labels.unique().tolist()))
    tpa_metrics = _metrics(labels, tpa_logits, class_ids, empty_index)
    vfep_rows: list[dict] = []
    vfep_results: dict[float, VfepResult] = {}
    for strength in strengths:
        result = build_vfep_logits(
            tpa_logits,
            metadata,
            sequence_field_index=sequence_field_index,
            strength=float(strength),
            empty_class_index=empty_index,
        )
        vfep_results[float(strength)] = result
        vfep_rows.append({"strength": float(strength), **_metrics(labels, result.logits, class_ids, empty_index)})
    selected_vfep = _select_strength(vfep_rows)
    selected_vfep_result = vfep_results[selected_vfep["strength"]]

    stp_rows: list[dict] = []
    stp_logits_by_strength: dict[float, torch.Tensor] = {}
    for strength in stp_strengths:
        logits = apply_sequence_consensus(tpa_logits, metadata, sequence_field_index, float(strength))
        stp_logits_by_strength[float(strength)] = logits
        stp_rows.append({"strength": float(strength), **_metrics(labels, logits, class_ids, empty_index)})
    selected_stp = _select_strength(stp_rows)
    selected_stp_logits = stp_logits_by_strength[selected_stp["strength"]]

    selected_strength = selected_vfep["strength"]
    control_results: dict[str, VfepResult] = {
        "uniform": build_vfep_logits(tpa_logits, metadata, sequence_field_index=sequence_field_index, strength=selected_strength, source_weighting="uniform"),
        "self_inclusive": build_vfep_logits(tpa_logits, metadata, sequence_field_index=sequence_field_index, strength=selected_strength, include_target=True),
        "arithmetic": build_vfep_logits(tpa_logits, metadata, sequence_field_index=sequence_field_index, strength=selected_strength, aggregation="arithmetic"),
        "all_class": build_vfep_logits(tpa_logits, metadata, sequence_field_index=sequence_field_index, strength=selected_strength, factorized=False),
        "location_prior": build_vfep_logits(
            tpa_logits,
            metadata,
            sequence_field_index=sequence_field_index,
            location_field_index=location_field_index,
            support_scope="location",
            strength=selected_strength,
        ),
    }
    control_metrics = {name: _metrics(labels, result.logits, class_ids, empty_index) for name, result in control_results.items()}

    shuffle_rows: list[dict] = []
    for seed in shuffle_seeds:
        shuffle = build_vfep_shuffle_groups(
            metadata,
            sequence_field_index=sequence_field_index,
            location_field_index=location_field_index,
            seed=int(seed),
        )
        shuffled = build_vfep_logits(
            tpa_logits,
            metadata,
            sequence_field_index=sequence_field_index,
            strength=selected_strength,
            groups=shuffle.groups,
        )
        eligible = torch.tensor(shuffle.available_frame_mask, dtype=torch.bool, device=labels.device)
        valid = bool(eligible.any().item()) and shuffle.original_pair_retention == 0.0
        if valid:
            stp_metrics = _metrics(labels[eligible], selected_stp_logits[eligible], class_ids, empty_index)
            real_metrics = _metrics(labels[eligible], selected_vfep_result.logits[eligible], class_ids, empty_index)
            shuffled_metrics = _metrics(labels[eligible], shuffled.logits[eligible], class_ids, empty_index)
            shuffled_gain = shuffled_metrics["macro_f1"] - stp_metrics["macro_f1"]
            real_gain = real_metrics["macro_f1"] - stp_metrics["macro_f1"]
        else:
            stp_metrics = None
            real_metrics = None
            shuffled_metrics = None
            shuffled_gain = None
            real_gain = None
        shuffle_rows.append({
            "seed": int(seed),
            "valid": valid,
            "eligible_frame_count": int(eligible.sum().item()),
            "eligible_frame_fraction": float(eligible.float().mean().item()),
            "stp_metrics_on_eligible": stp_metrics,
            "real_vfep_metrics_on_eligible": real_metrics,
            "shuffled_vfep_metrics_on_eligible": shuffled_metrics,
            "shuffled_gain_vs_stp_on_eligible": shuffled_gain,
            "real_gain_vs_stp_on_eligible": real_gain,
            "changed_frame_fraction": shuffle.changed_frame_fraction,
            "original_pair_retention": shuffle.original_pair_retention,
            "unavailable_group_count": shuffle.unavailable_group_count,
        })

    reference_predictions = selected_stp_logits.argmax(dim=1)
    vfep_predictions = selected_vfep_result.logits.argmax(dim=1)
    fixed_bootstrap = paired_fixed_class_location_bootstrap(
        labels,
        reference_predictions,
        vfep_predictions,
        location_keys,
        class_ids,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    legacy_bootstrap = paired_location_bootstrap(
        labels,
        reference_predictions,
        vfep_predictions,
        location_keys,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    real_gain = selected_vfep["macro_f1"] - selected_stp["macro_f1"]
    valid_shuffle_rows = [row for row in shuffle_rows if row["valid"]]
    shuffle_gains = [float(row["shuffled_gain_vs_stp_on_eligible"]) for row in valid_shuffle_rows]
    shuffle_gaps = [
        float(row["real_gain_vs_stp_on_eligible"]) - float(row["shuffled_gain_vs_stp_on_eligible"])
        for row in valid_shuffle_rows
    ]
    shuffle_control_valid = len(valid_shuffle_rows) == len(shuffle_rows) and bool(valid_shuffle_rows)
    shuffle_gap = median(shuffle_gaps) if shuffle_control_valid else None
    vfep_transitions = _species_transitions(labels, tpa_logits.argmax(dim=1), vfep_predictions, empty_index)
    stp_transitions = _species_transitions(labels, tpa_logits.argmax(dim=1), reference_predictions, empty_index)
    invariants = _visibility_invariants(labels, tpa_logits, selected_vfep_result.logits, empty_index)
    promotion = {
        "macro_f1_gain_at_least_0p30pp": real_gain >= 0.003,
        "fixed_location_bootstrap_lower_gt_zero": fixed_bootstrap.low > 0.0,
        "animal_macro_f1_non_regression": selected_vfep["animal_macro_f1"] >= selected_stp["animal_macro_f1"],
        "net_species_correction_better_than_stp": vfep_transitions["net"] > stp_transitions["net"],
        "shuffle_control_valid": shuffle_control_valid,
        "shuffle_gap_at_least_0p25pp": shuffle_gap is not None and shuffle_gap >= 0.0025,
        "visibility_invariants": invariants["empty_confusion_equal"] and invariants["max_empty_posterior_error"] <= 1e-6,
    }
    promotion["passed"] = all(promotion.values())
    report = {
        "protocol": "vfep_v0_val_audit_exploratory",
        "primary_metric": "fixed-supported-class F1-macro_all",
        "class_ids": list(class_ids),
        "tpa": tpa_metrics,
        "vfep_selection": vfep_rows,
        "selected_vfep": selected_vfep,
        "stp_selection": stp_rows,
        "selected_stp": selected_stp,
        "historical_stp_eta_0p5": next((row for row in stp_rows if row["strength"] == 0.5), None),
        "controls": control_metrics,
        "shuffles": shuffle_rows,
        "real_gain_vs_stp": real_gain,
        "shuffle_median_gain_vs_stp": median(shuffle_gains) if shuffle_gains else None,
        "real_minus_shuffle_median": shuffle_gap,
        "shuffle_control_valid": shuffle_control_valid,
        "fixed_class_location_bootstrap": asdict(fixed_bootstrap),
        "legacy_active_class_location_bootstrap": asdict(legacy_bootstrap),
        "equal_location_top1_sensitivity": _equal_location_top1(labels, reference_predictions, vfep_predictions, location_keys),
        "vfep_species_transitions": vfep_transitions,
        "stp_species_transitions": stp_transitions,
        "visibility_invariants": invariants,
        "corrected_frame_count": int((selected_vfep_result.propagation_strength > 0).sum().item()),
        "visibility_capped_frame_count": int(selected_vfep_result.visibility_capped.sum().item()),
        "mean_effective_lambda": float(selected_vfep_result.propagation_strength.mean().item()),
        "promotion": promotion,
    }
    logits = {
        "tpa": tpa_logits,
        "stp": selected_stp_logits,
        "vfep": selected_vfep_result.logits,
        **{name: result.logits for name, result in control_results.items()},
    }
    return report, logits
