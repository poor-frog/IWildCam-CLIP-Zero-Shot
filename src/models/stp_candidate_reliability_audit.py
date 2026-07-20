from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Sequence

import numpy as np
import torch

from src.models.loo_bcpd import sequence_groups, shuffled_sequence_groups
from src.models.stmp_adapter import metadata_group_key
from src.models.stp_audit_metrics import macro_f1, paired_location_bootstrap


FEATURE_NAMES = (
    "query_tpa_probability_for_candidate",
    "query_tpa_logit_for_candidate",
    "query_candidate_rank",
    "query_top1_margin",
    "sequence_mean_probability_for_candidate",
    "sequence_max_probability_for_candidate",
    "sequence_vote_fraction_for_candidate",
    "leave_one_out_vote_fraction_for_candidate",
    "leave_one_out_mean_probability_for_candidate",
    "stp_logit_for_candidate",
    "candidate_train_class_log_count",
    "candidate_is_empty",
    "sequence_length",
)


def location_fold_assignments(location_keys: Sequence[str], *, fold_count: int, seed: int) -> dict[str, int]:
    locations = sorted(
        set(location_keys),
        key=lambda location: hashlib.sha256(f"{seed}|{location}".encode("utf-8")).hexdigest(),
    )
    return {location: index % fold_count for index, location in enumerate(locations)}


def _group_membership(groups: Sequence[Sequence[int]], num_frames: int) -> tuple[tuple[int, ...], ...]:
    membership: list[tuple[int, ...] | None] = [None] * num_frames
    for group in groups:
        canonical = tuple(group)
        for index in group:
            if membership[index] is not None:
                raise ValueError("A frame belongs to multiple candidate-reliability groups.")
            membership[index] = canonical
    if any(group is None for group in membership):
        raise ValueError("Every frame must belong to a candidate-reliability group.")
    return tuple(group for group in membership if group is not None)


def _candidate_features(
    frame_index: int,
    candidate: int,
    group: Sequence[int],
    *,
    tpa_logits: torch.Tensor,
    tpa_probabilities: torch.Tensor,
    tpa_predictions: torch.Tensor,
    stp_logits: torch.Tensor,
    train_class_counts: torch.Tensor,
) -> tuple[float, ...]:
    indices = torch.tensor(group, dtype=torch.long)
    group_probabilities = tpa_probabilities[indices]
    group_predictions = tpa_predictions[indices]
    query_probabilities = tpa_probabilities[frame_index]
    query_order = torch.argsort(query_probabilities, descending=True, stable=True)
    query_rank = int(torch.where(query_order == candidate)[0][0].item()) + 1
    top_two = query_probabilities.topk(k=2).values
    query_margin = float((top_two[0] - top_two[1]).item())
    votes = int((group_predictions == candidate).sum().item())
    length = len(group)
    if length > 1:
        loo_votes = votes - int(tpa_predictions[frame_index].item() == candidate)
        loo_vote_fraction = loo_votes / (length - 1)
        loo_mean_probability = float(
            ((group_probabilities[:, candidate].sum() - query_probabilities[candidate]) / (length - 1)).item()
        )
    else:
        loo_vote_fraction = 0.0
        loo_mean_probability = 0.0
    return (
        float(query_probabilities[candidate].item()),
        float(tpa_logits[frame_index, candidate].item()),
        float(query_rank),
        query_margin,
        float(group_probabilities[:, candidate].mean().item()),
        float(group_probabilities[:, candidate].max().item()),
        votes / length,
        loo_vote_fraction,
        loo_mean_probability,
        float(stp_logits[frame_index, candidate].item()),
        math.log1p(float(train_class_counts[candidate].item())),
        float(candidate == 0),
        float(length),
    )


def build_candidate_rows(
    *,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    stp_logits: torch.Tensor,
    groups: Sequence[Sequence[int]],
    train_class_counts: torch.Tensor,
) -> dict[int, tuple[tuple[int, tuple[float, ...], int], ...]]:
    if tuple(FEATURE_NAMES) != FEATURE_NAMES:
        raise AssertionError("Feature names must be immutable.")
    probabilities = torch.softmax(tpa_logits.float(), dim=1)
    predictions = tpa_logits.argmax(dim=1)
    membership = _group_membership(groups, labels.shape[0])
    rows: dict[int, tuple[tuple[int, tuple[float, ...], int], ...]] = {}
    for frame_index, group in enumerate(membership):
        candidates = sorted(set(predictions[list(group)].tolist()))
        frame_rows = []
        for candidate in candidates:
            features = _candidate_features(
                frame_index,
                candidate,
                group,
                tpa_logits=tpa_logits,
                tpa_probabilities=probabilities,
                tpa_predictions=predictions,
                stp_logits=stp_logits,
                train_class_counts=train_class_counts,
            )
            frame_rows.append((candidate, features, int(candidate == int(labels[frame_index].item()))))
        rows[frame_index] = tuple(frame_rows)
    return rows


def _candidate_matrix(rows: dict[int, tuple[tuple[int, tuple[float, ...], int], ...]], frame_indices: Sequence[int]):
    records = [
        (frame, candidate, features, target)
        for frame in frame_indices
        if len(rows[frame]) > 1
        for candidate, features, target in rows[frame]
    ]
    if not records:
        raise ValueError("Candidate reliability requires at least one candidate row.")
    features = np.asarray([record[2] for record in records], dtype=np.float64)
    targets = np.asarray([record[3] for record in records], dtype=np.int64)
    return records, features, targets


def _fit_fold_predictions(
    rows: dict[int, tuple[tuple[int, tuple[float, ...], int], ...]],
    *,
    train_frames: Sequence[int],
    evaluation_frames: Sequence[int],
    stp_logits: torch.Tensor,
    stp_predictions: torch.Tensor,
    seed: int,
    permute_training_targets: bool = False,
) -> tuple[dict[int, int], dict[str, object]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train_records, train_features, train_targets = _candidate_matrix(rows, train_frames)
    if permute_training_targets:
        train_targets = np.random.default_rng(seed).permutation(train_targets)
    if np.unique(train_targets).size != 2:
        raise ValueError("Candidate-reliability training rows must include positive and negative targets.")
    scaler = StandardScaler().fit(train_features)
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="liblinear",
    ).fit(scaler.transform(train_features), train_targets)
    output: dict[int, int] = {}
    for frame_index in evaluation_frames:
        frame_rows = rows[frame_index]
        if len(frame_rows) <= 1:
            output[frame_index] = int(stp_predictions[frame_index].item())
            continue
        features = np.asarray([row[1] for row in frame_rows], dtype=np.float64)
        scores = model.predict_proba(scaler.transform(features))[:, 1]
        best_score = float(scores.max())
        tied = [position for position, score in enumerate(scores) if math.isclose(float(score), best_score, abs_tol=1e-12)]
        selected = min(
            tied,
            key=lambda position: (-float(stp_logits[frame_index, frame_rows[position][0]].item()), frame_rows[position][0]),
        )
        output[frame_index] = frame_rows[selected][0]
    diagnostics = {
        "training_candidate_row_count": len(train_records),
        "training_positive_fraction": float(train_targets.mean()),
        "coefficient_l2_norm": float(np.linalg.norm(model.coef_)),
        "standardized_coefficients": {
            name: float(value) for name, value in zip(FEATURE_NAMES, model.coef_[0], strict=True)
        },
        "intercept": float(model.intercept_[0]),
    }
    return output, diagnostics


def _run_oof_selector(
    *,
    labels: torch.Tensor,
    rows: dict[int, tuple[tuple[int, tuple[float, ...], int], ...]],
    location_keys: Sequence[str],
    fold_assignments: dict[str, int],
    stp_logits: torch.Tensor,
    seed: int,
    permute_training_targets: bool = False,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    stp_predictions = stp_logits.argmax(dim=1)
    predictions = stp_predictions.clone()
    fold_records = []
    for fold in range(5):
        evaluation_frames = [index for index, location in enumerate(location_keys) if fold_assignments[location] == fold]
        train_frames = [index for index, location in enumerate(location_keys) if fold_assignments[location] != fold]
        evaluation_locations = sorted({location_keys[index] for index in evaluation_frames})
        supported_classes = int(labels[evaluation_frames].unique().numel())
        viable = len(evaluation_locations) >= 4 and len(evaluation_frames) >= 100 and supported_classes >= 5
        fold_record = {
            "fold": fold,
            "evaluation_location_digests": [hashlib.sha256(location.encode("utf-8")).hexdigest() for location in evaluation_locations],
            "evaluation_location_count": len(evaluation_locations),
            "evaluation_frame_count": len(evaluation_frames),
            "evaluation_supported_class_count": supported_classes,
            "viability_pass": viable,
        }
        if not viable:
            fold_records.append(fold_record)
            continue
        try:
            fold_predictions, diagnostics = _fit_fold_predictions(
                rows,
                train_frames=train_frames,
                evaluation_frames=evaluation_frames,
                stp_logits=stp_logits,
                stp_predictions=stp_predictions,
            seed=seed + fold if permute_training_targets else seed,
                permute_training_targets=permute_training_targets,
            )
        except ValueError as error:
            fold_record["viability_pass"] = False
            fold_record["candidate_training_viability_error"] = str(error)
            fold_records.append(fold_record)
            continue
        for frame_index, prediction in fold_predictions.items():
            predictions[frame_index] = prediction
        fold_record.update(diagnostics)
        evaluation_tensor = torch.tensor(evaluation_frames, dtype=torch.long)
        fold_record["descriptive_metrics"] = {
            "stp_macro_f1": macro_f1(
                labels[evaluation_tensor], stp_predictions[evaluation_tensor], stp_logits.shape[1]
            ),
            "selector_macro_f1": macro_f1(
                labels[evaluation_tensor], predictions[evaluation_tensor], stp_logits.shape[1]
            ),
            "descriptive_only": True,
        }
        fold_records.append(fold_record)
    return predictions, fold_records


def _method_metrics(labels: torch.Tensor, predictions: torch.Tensor, train_class_counts: torch.Tensor) -> dict[str, object]:
    num_classes = train_class_counts.numel()
    animal_mask = labels != 0
    tail_mask = train_class_counts[labels] <= 20
    empty_to_animal = int(((labels == 0) & (predictions != 0)).sum().item())
    animal_to_empty = int(((labels != 0) & (predictions == 0)).sum().item())
    return {
        "macro_f1": macro_f1(labels, predictions, num_classes),
        "top1": float((labels == predictions).float().mean().item()),
        "animal_macro_f1": macro_f1(labels[animal_mask], predictions[animal_mask], num_classes),
        "tail_macro_f1": macro_f1(labels[tail_mask], predictions[tail_mask], num_classes),
        "empty_to_animal_error_count": empty_to_animal,
        "animal_to_empty_error_count": animal_to_empty,
        "empty_animal_total_error_count": empty_to_animal + animal_to_empty,
    }


def _simple_candidate_baselines(
    *,
    tpa_logits: torch.Tensor,
    stp_logits: torch.Tensor,
    groups: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = torch.softmax(tpa_logits.float(), dim=1)
    tpa_predictions = tpa_logits.argmax(dim=1)
    stp_predictions = stp_logits.argmax(dim=1)
    plurality = stp_predictions.clone()
    mean_probability = stp_predictions.clone()
    for group in groups:
        if len(group) <= 1:
            continue
        indices = torch.tensor(group, dtype=torch.long)
        candidates = sorted(set(tpa_predictions[indices].tolist()))
        for frame_index in group:
            plurality[frame_index] = min(
                candidates,
                key=lambda candidate: (
                    -int((tpa_predictions[indices] == candidate).sum().item()),
                    -float(stp_logits[frame_index, candidate].item()),
                    candidate,
                ),
            )
            mean_probability[frame_index] = min(
                candidates,
                key=lambda candidate: (-float(probabilities[indices, candidate].mean().item()), candidate),
            )
    return plurality, mean_probability


def build_candidate_reliability_audit_report(
    *,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    stp_logits: torch.Tensor,
    metadata: Sequence[torch.Tensor],
    location_keys: Sequence[str],
    sequence_field_index: int,
    location_field_index: int,
    train_class_counts: torch.Tensor,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 20260721,
    shuffle_seeds: Sequence[int] = tuple(range(20260721, 20260741)),
    permutation_seeds: Sequence[int] = tuple(range(20260741, 20260761)),
) -> tuple[dict[str, object], dict[str, object]]:
    if len(metadata) != labels.shape[0] or len(location_keys) != labels.shape[0]:
        raise ValueError("Candidate-reliability inputs must describe the same Val-Audit frames.")
    metadata_locations = tuple(metadata_group_key(row, location_field_index) for row in metadata)
    if any(location is None for location in metadata_locations) or tuple(str(location) for location in metadata_locations) != tuple(location_keys):
        raise ValueError("Candidate-reliability location keys must match the supplied metadata.")
    if len(set(location_keys)) != 20:
        raise ValueError("Candidate Reliability Audit v0 locks five folds over exactly 20 Val-Audit locations.")
    if len(shuffle_seeds) != 20 or len(permutation_seeds) != 20:
        raise ValueError("Candidate Reliability Audit v0 locks both negative controls to 20 seeds.")
    folds = location_fold_assignments(location_keys, fold_count=5, seed=20260721)
    real_groups = sequence_groups(metadata, sequence_field_index, labels.shape[0])
    real_rows = build_candidate_rows(
        labels=labels,
        tpa_logits=tpa_logits,
        stp_logits=stp_logits,
        groups=real_groups,
        train_class_counts=train_class_counts,
    )
    selector_predictions, fold_records = _run_oof_selector(
        labels=labels,
        rows=real_rows,
        location_keys=location_keys,
        fold_assignments=folds,
        stp_logits=stp_logits,
        seed=20260721,
    )
    fold_manifest = {
        "fold_seed": 20260721,
        "assignment": "sha256_sorted_round_robin",
        "folds": fold_records,
        "all_folds_viable": all(record["viability_pass"] for record in fold_records),
        "raw_location_ids_materialized": False,
    }
    if not fold_manifest["all_folds_viable"]:
        return {
            "audit": "stp_candidate_reliability_audit_v0",
            "status": "blocked_fold_viability",
            "promotion_decision_materialized": False,
        }, fold_manifest

    stp_predictions = stp_logits.argmax(dim=1)
    tpa_predictions = tpa_logits.argmax(dim=1)
    plurality_predictions, mean_probability_predictions = _simple_candidate_baselines(
        tpa_logits=tpa_logits,
        stp_logits=stp_logits,
        groups=real_groups,
    )
    metrics = {
        "tpa": _method_metrics(labels, tpa_predictions, train_class_counts),
        "stp": _method_metrics(labels, stp_predictions, train_class_counts),
        "sequence_plurality": _method_metrics(labels, plurality_predictions, train_class_counts),
        "sequence_mean_probability": _method_metrics(labels, mean_probability_predictions, train_class_counts),
        "diagnostic_selector_oof": _method_metrics(labels, selector_predictions, train_class_counts),
    }
    bootstrap = asdict(paired_location_bootstrap(
        labels,
        stp_predictions,
        selector_predictions,
        location_keys,
        bootstrap_samples=bootstrap_samples,
        seed=bootstrap_seed,
    ))
    real_gain = metrics["diagnostic_selector_oof"]["macro_f1"] - metrics["stp"]["macro_f1"]

    shuffle_records = []
    controls_all_viable = True
    for seed in shuffle_seeds:
        shuffled = shuffled_sequence_groups(metadata, sequence_field_index, location_field_index, None, seed)
        shuffled_rows = build_candidate_rows(
            labels=labels,
            tpa_logits=tpa_logits,
            stp_logits=stp_logits,
            groups=shuffled.groups,
            train_class_counts=train_class_counts,
        )
        predictions, control_folds = _run_oof_selector(
            labels=labels,
            rows=shuffled_rows,
            location_keys=location_keys,
            fold_assignments=folds,
            stp_logits=stp_logits,
            seed=20260721,
        )
        if not all(record["viability_pass"] for record in control_folds):
            controls_all_viable = False
            continue
        gain = _method_metrics(labels, predictions, train_class_counts)["macro_f1"] - metrics["stp"]["macro_f1"]
        shuffle_records.append({"seed": seed, "gain_vs_stp": gain, "changed_frame_fraction": shuffled.changed_frame_fraction})

    permutation_records = []
    for seed in permutation_seeds:
        predictions, control_folds = _run_oof_selector(
            labels=labels,
            rows=real_rows,
            location_keys=location_keys,
            fold_assignments=folds,
            stp_logits=stp_logits,
            seed=seed,
            permute_training_targets=True,
        )
        if not all(record["viability_pass"] for record in control_folds):
            controls_all_viable = False
            continue
        gain = _method_metrics(labels, predictions, train_class_counts)["macro_f1"] - metrics["stp"]["macro_f1"]
        permutation_records.append({"seed": seed, "gain_vs_stp": gain})

    shuffle_median = median(record["gain_vs_stp"] for record in shuffle_records) if shuffle_records else None
    permutation_median = median(record["gain_vs_stp"] for record in permutation_records) if permutation_records else None
    selector = metrics["diagnostic_selector_oof"]
    stp = metrics["stp"]
    conditions = {
        "macro_f1_gain_at_least_3pp": real_gain >= 0.03,
        "location_bootstrap_lower_bound_gt_zero": bootstrap["low"] > 0.0,
        "animal_macro_f1_non_regression": selector["animal_macro_f1"] >= stp["animal_macro_f1"],
        "tail_macro_f1_within_0p5pp": selector["tail_macro_f1"] >= stp["tail_macro_f1"] - 0.005,
        "empty_animal_total_error_non_increase": selector["empty_animal_total_error_count"] <= stp["empty_animal_total_error_count"],
        "real_gain_minus_shuffle_median_at_least_1pp": shuffle_median is not None and real_gain - shuffle_median >= 0.01,
        "permuted_target_gain_median_at_most_0p5pp": permutation_median is not None and permutation_median <= 0.005,
    }
    passed = controls_all_viable and all(conditions.values())
    if not controls_all_viable:
        outcome = "inconclusive_repair_audit"
    elif passed:
        outcome = "authorize_one_candidate_reranker_v0"
    else:
        outcome = "close_all_sequence_inference_development"
    return {
        "audit": "stp_candidate_reliability_audit_v0",
        "status": "complete",
        "split": "IWildCam Val-Audit",
        "deployable_method_result": False,
        "feature_names": list(FEATURE_NAMES),
        "metrics": metrics,
        "oof_prediction_count": int(selector_predictions.numel()),
        "oof_predictions": selector_predictions.tolist(),
        "primary": {
            "gain_vs_stp": real_gain,
            "location_bootstrap": bootstrap,
            "sequence_candidate_oracle_headroom_captured_fraction": real_gain / 0.1581813395023346,
        },
        "negative_controls": {
            "all_controls_viable": controls_all_viable,
            "within_location_sequence_shuffle": {"records": shuffle_records, "median_gain_vs_stp": shuffle_median},
            "training_target_permutation": {"records": permutation_records, "median_gain_vs_stp": permutation_median},
        },
        "promotion": {
            "conditions": conditions,
            "passed": passed,
            "outcome": outcome,
        },
        "firewall": {
            "val_confirm_predictions_materialized": False,
            "iwildcam_ood_predictions_materialized": False,
            "cct20_opened": False,
        },
    }, fold_manifest


def write_candidate_reliability_artifacts(
    output_dir: Path,
    *,
    preregistration_path: Path,
    fold_manifest: dict[str, object],
    class_mapping: dict[str, object],
    report: dict[str, object],
) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty candidate-reliability directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "preregistration.json": json.loads(preregistration_path.read_text(encoding="utf-8")),
        "fold_manifest.json": fold_manifest,
        "class_mapping.json": class_mapping,
        "candidate_reliability_audit_v0.json": report,
    }
    for filename, payload in payloads.items():
        (output_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report["status"] == "complete":
        primary = report["primary"]
        lines = [
            "# STP Candidate Reliability Audit v0",
            "",
            "Development-only pooled out-of-fold diagnostic. No held-out split was opened.",
            "",
            f"- STP macro-F1: {report['metrics']['stp']['macro_f1'] * 100:.2f}%",
            f"- Diagnostic selector macro-F1: {report['metrics']['diagnostic_selector_oof']['macro_f1'] * 100:.2f}%",
            f"- Gain: {primary['gain_vs_stp'] * 100:+.2f} pp",
            f"- Location-bootstrap 95% CI: [{primary['location_bootstrap']['low'] * 100:+.2f}, {primary['location_bootstrap']['high'] * 100:+.2f}] pp",
            f"- Frozen decision: `{report['promotion']['outcome']}`",
        ]
    else:
        lines = ["# STP Candidate Reliability Audit v0", "", "Blocked by preregistered fold viability; no promotion decision was materialized."]
    (output_dir / "candidate_reliability_audit_v0.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.iterdir())
    }
    receipt = {
        "audit": "stp_candidate_reliability_audit_v0",
        "artifact_sha256": hashes,
        "immutable_output_policy": "refuse_nonempty_directory",
        "val_confirm_predictions_materialized": False,
        "iwildcam_ood_predictions_materialized": False,
        "cct20_opened": False,
    }
    (output_dir / "audit_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
