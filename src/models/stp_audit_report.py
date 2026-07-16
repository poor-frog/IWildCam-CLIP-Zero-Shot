from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import torch

from src.models.loo_bcpd import sequence_groups
from src.models.stmp_adapter import metadata_value
from src.models.stp_audit_metrics import comparison_payload


def audit_bin_edges(logits: torch.Tensor) -> dict[str, list[float]]:
    probabilities = torch.softmax(logits.float(), dim=1)
    confidence = probabilities.max(dim=1).values
    top_two = probabilities.topk(k=2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    return {
        "confidence": torch.quantile(confidence, torch.tensor([0.25, 0.50, 0.75])).tolist(),
        "margin": torch.quantile(margin, torch.tensor([0.25, 0.50, 0.75])).tolist(),
    }


def _sequence_key(metadata_row, sequence_field_index: int | None) -> str | None:
    value = metadata_value(metadata_row, sequence_field_index)
    if value is None:
        return None
    if isinstance(value, float) and not torch.isfinite(torch.tensor(value)):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    value = str(value).strip()
    return value or None


def _taxonomy(labels: torch.Tensor, groups: Sequence[Sequence[int]], metadata, sequence_field_index: int | None) -> list[str]:
    values = ["unknown"] * labels.shape[0]
    for group in groups:
        if _sequence_key(metadata[group[0]], sequence_field_index) is None:
            continue
        group_labels = labels[list(group)]
        non_empty = torch.unique(group_labels[group_labels != 0])
        if non_empty.numel() == 0:
            name = "all_empty"
        elif non_empty.numel() >= 2:
            name = "multi_species"
        elif (group_labels == 0).any():
            name = "one_species_mixed_empty"
        else:
            name = "one_species_no_empty"
        for index in group:
            values[index] = name
    return values


def _length_bin(length: int) -> str:
    if length <= 1:
        return "1"
    if length == 2:
        return "2"
    if length <= 5:
        return "3_5"
    if length <= 10:
        return "6_10"
    return "11_plus"


def _frequency_bin(count: int) -> str:
    if count <= 20:
        return "tail"
    if count <= 100:
        return "medium"
    return "head"


def _quartile_bins(values: torch.Tensor, edges: Sequence[float]) -> list[str]:
    first, second, third = (float(edge) for edge in edges)
    labels = []
    for value in values.tolist():
        if value <= first:
            labels.append("q1")
        elif value <= second:
            labels.append("q2")
        elif value <= third:
            labels.append("q3")
        else:
            labels.append("q4")
    return labels


def _sequence_cluster_keys(groups: Sequence[Sequence[int]], num_examples: int) -> tuple[str, ...]:
    keys = [""] * num_examples
    for group_index, group in enumerate(groups):
        for index in group:
            keys[index] = f"sequence:{group_index}"
    if any(not key for key in keys):
        raise ValueError("Every audit frame must belong to exactly one sequence bootstrap cluster.")
    return tuple(keys)


def sequence_context_statistics(logits: torch.Tensor, metadata, sequence_field_index: int | None) -> dict[str, torch.Tensor]:
    groups = sequence_groups(metadata, sequence_field_index, logits.shape[0])
    probabilities = torch.softmax(logits.float(), dim=1)
    predictions = logits.argmax(dim=1)
    agreement = torch.ones(logits.shape[0])
    concentration = torch.zeros(logits.shape[0])
    lengths = torch.ones(logits.shape[0], dtype=torch.long)
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long)
        group_predictions = predictions[indices]
        modal_count = torch.bincount(group_predictions, minlength=logits.shape[1]).max()
        mean_probability = probabilities[indices].mean(dim=0)
        entropy = -(mean_probability * mean_probability.clamp_min(1e-12).log()).sum()
        agreement[indices] = modal_count.float() / len(group)
        concentration[indices] = 1.0 - entropy / torch.log(torch.tensor(float(logits.shape[1])))
        lengths[indices] = len(group)
    return {"agreement": agreement, "posterior_concentration": concentration, "length": lengths}


def _stratified_payloads(
    labels: torch.Tensor,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    location_keys: Sequence[str],
    train_class_counts: torch.Tensor,
    strata: dict[str, Sequence[str]],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    output: dict[str, object] = {}
    for label, values in strata.items():
        indices = torch.tensor([index for index, value in enumerate(values) if value == label], dtype=torch.long)
        if indices.numel() == 0:
            continue
        unique_locations = len({location_keys[index] for index in indices.tolist()})
        supported_classes = int(labels[indices].unique().numel())
        record: dict[str, object] = {
            "frame_count": int(indices.numel()),
            "location_count": unique_locations,
            "supported_class_count": supported_classes,
            "descriptive_only": unique_locations < 10 or supported_classes < 5,
        }
        try:
            record["comparison"] = comparison_payload(
                labels[indices], reference_logits[indices], candidate_logits[indices],
                [location_keys[index] for index in indices.tolist()], train_class_counts,
                bootstrap_samples=bootstrap_samples, seed=seed,
            )
        except ValueError as error:
            record["comparison_error"] = str(error)
        output[label] = record
    return output


def build_mechanism_audit_report(
    *,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    stp_mean_logits: torch.Tensor,
    stp_loo_logits: torch.Tensor,
    metadata,
    location_keys: Sequence[str],
    sequence_field_index: int | None,
    train_class_counts: torch.Tensor,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    groups = sequence_groups(metadata, sequence_field_index, labels.shape[0])
    sequence_keys = _sequence_cluster_keys(groups, labels.shape[0])
    taxonomy = _taxonomy(labels, groups, metadata, sequence_field_index)
    context = sequence_context_statistics(tpa_logits, metadata, sequence_field_index)
    length_bins = [_length_bin(int(length)) for length in context["length"].tolist()]
    frequency_bins = [_frequency_bin(int(train_class_counts[label].item())) for label in labels.tolist()]
    edges = audit_bin_edges(tpa_logits)
    probabilities = torch.softmax(tpa_logits.float(), dim=1)
    confidence = probabilities.max(dim=1).values
    top_two = probabilities.topk(k=2, dim=1).values
    margin = top_two[:, 0] - top_two[:, 1]
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    base = {
        "tpa_to_stp_mean": comparison_payload(
            labels, tpa_logits, stp_mean_logits, location_keys, train_class_counts,
            bootstrap_samples=bootstrap_samples, seed=seed, sequence_keys=sequence_keys,
        ),
        "tpa_to_stp_loo": comparison_payload(
            labels, tpa_logits, stp_loo_logits, location_keys, train_class_counts,
            bootstrap_samples=bootstrap_samples, seed=seed + 1, sequence_keys=sequence_keys,
        ),
        "stp_mean_to_stp_loo": comparison_payload(
            labels, stp_mean_logits, stp_loo_logits, location_keys, train_class_counts,
            bootstrap_samples=bootstrap_samples, seed=seed + 2, sequence_keys=sequence_keys,
        ),
    }
    strata = {
        "taxonomy": taxonomy,
        "burst_length": length_bins,
        "true_frequency": frequency_bins,
        "confidence_quartile": _quartile_bins(confidence, edges["confidence"]),
        "margin_quartile": _quartile_bins(margin, edges["margin"]),
    }
    for name, values in strata.items():
        base[f"strata_{name}"] = {
            "tpa_to_stp_mean": _stratified_payloads(labels, tpa_logits, stp_mean_logits, location_keys, train_class_counts, {value: values for value in sorted(set(values))}, bootstrap_samples, seed),
            "tpa_to_stp_loo": _stratified_payloads(labels, tpa_logits, stp_loo_logits, location_keys, train_class_counts, {value: values for value in sorted(set(values))}, bootstrap_samples, seed + 1),
        }
    base["context"] = {
        "context_eligible_frame_fraction": float((context["length"] >= 2).float().mean().item()),
        "agreement_median": float(context["agreement"].median().item()),
        "posterior_concentration_median": float(context["posterior_concentration"].median().item()),
        "confidence_median": float(confidence.median().item()),
        "margin_median": float(margin.median().item()),
        "entropy_median": float(entropy.median().item()),
        "burst_length_counts": dict(sorted(Counter(length_bins).items())),
        "label_taxonomy_counts": dict(sorted(Counter(taxonomy).items())),
    }
    base["audit_bin_edges"] = edges
    return base


def write_audit_artifacts(output_dir: Path, manifest: dict[str, object], viability: dict[str, object], report: dict[str, object], class_mapping: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "audit_manifest.json": manifest,
        "confirm_viability_aggregate.json": viability,
        "class_mapping.json": class_mapping,
        "stp_mechanism_audit.json": report,
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# STP Mechanism Audit: Val-Audit", "", "Phase A only. No Val-Confirm prediction or performance artifact was created.", ""]
    for name in ("tpa_to_stp_mean", "tpa_to_stp_loo", "stp_mean_to_stp_loo"):
        comparison = report[name]
        lines.append(f"## {name}")
        lines.append(f"- Reference macro-F1: {comparison['reference_macro_f1'] * 100:.2f}%")
        lines.append(f"- Candidate macro-F1: {comparison['candidate_macro_f1'] * 100:.2f}%")
        lines.append(f"- Location-bootstrap delta: {comparison['location_bootstrap']['delta'] * 100:.2f} pp")
        lines.append("")
    (output_dir / "stp_mechanism_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
