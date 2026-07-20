from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Sequence

import torch

from src.models.loo_bcpd import sequence_groups, shuffled_sequence_groups
from src.models.stmp_adapter import metadata_group_key
from src.models.stp_audit_metrics import macro_f1, paired_location_bootstrap


ORACLE_AUDIT_VERSION = "stp_oracle_audit_v0"


def _prediction_metrics(labels: torch.Tensor, predictions: torch.Tensor, num_classes: int) -> dict[str, float]:
    return {
        "macro_f1": macro_f1(labels, predictions, num_classes),
        "top1": float((predictions == labels).float().mean().item()),
    }


def _apply_group_mean(logits: torch.Tensor, groups: Sequence[Sequence[int]], eta: float = 0.5) -> torch.Tensor:
    output = logits.clone()
    for group in groups:
        if len(group) <= 1:
            continue
        indices = torch.tensor(group, dtype=torch.long, device=logits.device)
        values = logits.index_select(0, indices)
        output[indices] = (1.0 - eta) * values + eta * values.mean(dim=0, keepdim=True)
    return output


def method_selection_oracle(
    labels: torch.Tensor,
    fallback_predictions: torch.Tensor,
    candidate_predictions: Sequence[torch.Tensor],
) -> torch.Tensor:
    output = fallback_predictions.clone()
    unresolved = output != labels
    for predictions in candidate_predictions:
        take = unresolved & (predictions == labels)
        output[take] = predictions[take]
        unresolved &= ~take
    return output


def sequence_candidate_oracle(
    labels: torch.Tensor,
    fallback_predictions: torch.Tensor,
    tpa_predictions: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    output = fallback_predictions.clone()
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long)
        candidates = set(tpa_predictions[indices].tolist())
        recoverable = torch.tensor([int(labels[index].item()) in candidates for index in group], dtype=torch.bool)
        selected = indices[recoverable]
        output[selected] = labels[selected]
    return output


def event_constant_oracle(
    labels: torch.Tensor,
    fallback_predictions: torch.Tensor,
    groups: Sequence[Sequence[int]],
    num_classes: int,
) -> torch.Tensor:
    output = fallback_predictions.clone()
    for group in groups:
        if len(group) <= 1:
            continue
        indices = torch.tensor(group, dtype=torch.long)
        modal_label = int(torch.bincount(labels[indices], minlength=num_classes).argmax().item())
        output[indices] = modal_label
    return output


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


def _quartile_bins(values: torch.Tensor) -> list[str]:
    edges = torch.quantile(values.float(), torch.tensor([0.25, 0.5, 0.75]))
    return [f"q{int(torch.bucketize(value, edges, right=False).item()) + 1}" for value in values]


def _taxonomy(labels: torch.Tensor, groups: Sequence[Sequence[int]]) -> list[str]:
    result = ["unknown"] * labels.shape[0]
    for group in groups:
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
            result[index] = name
    return result


def _strata_payload(
    labels: torch.Tensor,
    reference: torch.Tensor,
    oracle: torch.Tensor,
    values: Sequence[str],
    num_classes: int,
) -> dict[str, object]:
    payload: dict[str, object] = {}
    for value in sorted(set(values)):
        indices = torch.tensor([index for index, item in enumerate(values) if item == value], dtype=torch.long)
        reference_f1 = macro_f1(labels[indices], reference[indices], num_classes)
        oracle_f1 = macro_f1(labels[indices], oracle[indices], num_classes)
        payload[value] = {
            "frame_count": int(indices.numel()),
            "supported_class_count": int(labels[indices].unique().numel()),
            "stp_macro_f1": reference_f1,
            "oracle_macro_f1": oracle_f1,
            "headroom": oracle_f1 - reference_f1,
        }
    return payload


def _oracle_comparison(
    labels: torch.Tensor,
    reference: torch.Tensor,
    oracle: torch.Tensor,
    location_keys: Sequence[str],
    num_classes: int,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, object]:
    reference_metrics = _prediction_metrics(labels, reference, num_classes)
    oracle_metrics = _prediction_metrics(labels, oracle, num_classes)
    bootstrap = paired_location_bootstrap(
        labels,
        reference,
        oracle,
        location_keys,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "reference": reference_metrics,
        "oracle": oracle_metrics,
        "macro_f1_headroom": oracle_metrics["macro_f1"] - reference_metrics["macro_f1"],
        "recoverable_error_count": int(((reference != labels) & (oracle == labels)).sum().item()),
        "introduced_error_count": int(((reference == labels) & (oracle != labels)).sum().item()),
        "location_bootstrap": asdict(bootstrap),
    }


def _decision(primary_headroom: float) -> str:
    if primary_headroom < 0.03:
        return "stop_new_sequence_aggregators"
    if primary_headroom >= 0.05:
        return "targeted_method_only"
    return "inconclusive"


def build_oracle_audit_report(
    *,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    stp_mean_logits: torch.Tensor,
    stp_loo_logits: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    location_keys: Sequence[str],
    sequence_field_index: int,
    location_field_index: int,
    train_class_counts: torch.Tensor,
    bootstrap_samples: int,
    bootstrap_seed: int,
    shuffle_seeds: Sequence[int],
) -> dict[str, object]:
    if labels.ndim != 1 or len(metadata) != labels.shape[0] or len(location_keys) != labels.shape[0]:
        raise ValueError("Oracle audit inputs must describe the same Val-Audit frames.")
    if not shuffle_seeds:
        raise ValueError("Oracle audit requires at least one predeclared shuffle seed.")
    metadata_locations = tuple(metadata_group_key(row, location_field_index) for row in metadata)
    if any(value is None for value in metadata_locations) or tuple(str(value) for value in metadata_locations) != tuple(location_keys):
        raise ValueError("Location keys do not match the supplied Val-Audit metadata.")

    num_classes = tpa_logits.shape[1]
    groups = sequence_groups(metadata, sequence_field_index, labels.shape[0])
    tpa_predictions = tpa_logits.argmax(dim=1)
    stp_predictions = stp_mean_logits.argmax(dim=1)
    loo_predictions = stp_loo_logits.argmax(dim=1)
    method_oracle = method_selection_oracle(labels, stp_predictions, (tpa_predictions, loo_predictions))
    candidate_oracle = sequence_candidate_oracle(labels, stp_predictions, tpa_predictions, groups)
    constant_oracle = event_constant_oracle(labels, stp_predictions, groups, num_classes)

    comparisons = {
        "method_selection": _oracle_comparison(labels, stp_predictions, method_oracle, location_keys, num_classes, bootstrap_samples, bootstrap_seed),
        "sequence_candidate": _oracle_comparison(labels, stp_predictions, candidate_oracle, location_keys, num_classes, bootstrap_samples, bootstrap_seed + 1),
        "event_constant": _oracle_comparison(labels, stp_predictions, constant_oracle, location_keys, num_classes, bootstrap_samples, bootstrap_seed + 2),
    }

    shuffled_records = []
    for seed in shuffle_seeds:
        shuffled = shuffled_sequence_groups(metadata, sequence_field_index, location_field_index, None, seed)
        shuffled_logits = _apply_group_mean(tpa_logits, shuffled.groups, eta=0.5)
        metrics = _prediction_metrics(labels, shuffled_logits.argmax(dim=1), num_classes)
        shuffled_records.append({
            "seed": seed,
            "macro_f1": metrics["macro_f1"],
            "top1": metrics["top1"],
            "changed_frame_fraction": shuffled.changed_frame_fraction,
            "unavailable_group_count": shuffled.unavailable_group_count,
        })
    shuffled_f1 = [record["macro_f1"] for record in shuffled_records]
    stp_f1 = comparisons["method_selection"]["reference"]["macro_f1"]

    lengths = [1] * labels.shape[0]
    for group in groups:
        for index in group:
            lengths[index] = len(group)
    confidence = torch.softmax(tpa_logits.float(), dim=1).max(dim=1).values
    strata = {
        "burst_length": [_length_bin(length) for length in lengths],
        "sequence_label_taxonomy": _taxonomy(labels, groups),
        "true_class_frequency": [_frequency_bin(int(train_class_counts[label].item())) for label in labels.tolist()],
        "tpa_confidence_quartile": _quartile_bins(confidence),
    }

    per_location = []
    for location in sorted(set(location_keys)):
        indices = torch.tensor([index for index, value in enumerate(location_keys) if value == location], dtype=torch.long)
        reference_f1 = macro_f1(labels[indices], stp_predictions[indices], num_classes)
        oracle_f1 = macro_f1(labels[indices], method_oracle[indices], num_classes)
        per_location.append({
            "location_digest": hashlib.sha256(location.encode("utf-8")).hexdigest(),
            "frame_count": int(indices.numel()),
            "stp_macro_f1": reference_f1,
            "method_selection_oracle_macro_f1": oracle_f1,
            "headroom": oracle_f1 - reference_f1,
        })

    primary_headroom = comparisons["method_selection"]["macro_f1_headroom"]
    return {
        "audit": ORACLE_AUDIT_VERSION,
        "split": "IWildCam Val-Audit",
        "diagnostic_labels_used": True,
        "deployable_method_result": False,
        "reference": {"name": "stp_mean_eta_0.5", **_prediction_metrics(labels, stp_predictions, num_classes)},
        "oracles": comparisons,
        "negative_control": {
            "name": "within_location_shuffled_sequence_consensus",
            "records": shuffled_records,
            "median_macro_f1": median(shuffled_f1),
            "real_stp_minus_shuffle_median": stp_f1 - median(shuffled_f1),
        },
        "strata_method_selection_headroom": {
            name: _strata_payload(labels, stp_predictions, method_oracle, values, num_classes)
            for name, values in strata.items()
        },
        "per_location_method_selection_headroom": per_location,
        "counts": {
            "frames": int(labels.numel()),
            "locations": len(set(location_keys)),
            "sequences": len(groups),
            "burst_length_counts": dict(sorted(Counter(_length_bin(length) for length in lengths).items())),
        },
        "decision": {
            "primary_headroom": primary_headroom,
            "outcome": _decision(primary_headroom),
            "thresholds": {"stop_below": 0.03, "targeted_method_at_or_above": 0.05},
            "scope": "development_only_no_held_out_claim",
        },
    }


def write_oracle_audit_artifacts(
    output_dir: Path,
    *,
    preregistration_path: Path,
    manifest: dict[str, object],
    class_mapping: dict[str, object],
    report: dict[str, object],
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty oracle audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    payloads = {
        "preregistration.json": preregistration,
        "audit_manifest.json": manifest,
        "class_mapping.json": class_mapping,
        "stp_oracle_audit_v0.json": report,
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# STP Oracle Audit v0",
        "",
        "Development-only Val-Audit error decomposition. Oracle labels are diagnostic; these are not deployable results.",
        "Val-Confirm, ID, OOD, and CCT-20 predictions were not materialized.",
        "",
        f"- STP macro-F1: {report['reference']['macro_f1'] * 100:.2f}%",
        f"- Method-selection oracle headroom: {report['oracles']['method_selection']['macro_f1_headroom'] * 100:+.2f} pp",
        f"- Sequence-candidate oracle headroom: {report['oracles']['sequence_candidate']['macro_f1_headroom'] * 100:+.2f} pp",
        f"- Event-constant oracle headroom: {report['oracles']['event_constant']['macro_f1_headroom'] * 100:+.2f} pp",
        f"- Real STP minus shuffled median: {report['negative_control']['real_stp_minus_shuffle_median'] * 100:+.2f} pp",
        f"- Frozen decision: `{report['decision']['outcome']}`",
    ]
    (output_dir / "stp_oracle_audit_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_hashes = {}
    for path in sorted(output_dir.iterdir()):
        artifact_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "audit": ORACLE_AUDIT_VERSION,
        "artifact_sha256": artifact_hashes,
        "diagnostic_labels_opened_on_val_audit": True,
        "val_confirm_predictions_materialized": False,
        "iwildcam_ood_predictions_materialized": False,
        "cct20_opened": False,
        "immutable_output_policy": "refuse_nonempty_directory",
    }
    (output_dir / "oracle_audit_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
