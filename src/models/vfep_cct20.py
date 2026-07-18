from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


CCT_PROTOTYPE_SCALE = 50.0
CCT_EXPECTED_TRANS_VALIDATION_LOCATIONS = 1
CCT_EXPECTED_TRANS_TEST_LOCATIONS = 9


def canonical_cct_mapping(categories: Sequence[Mapping]) -> tuple[dict[int, int], tuple[str, ...], str]:
    normalized: list[tuple[int, str]] = []
    for category in categories:
        category_id = int(category["id"])
        name = str(category["name"]).strip().lower()
        if not name:
            raise ValueError("CCT categories must have non-empty names.")
        normalized.append((category_id, name))
    empty = [item for item in normalized if item[1] == "empty"]
    if len(empty) != 1:
        raise ValueError("CCT mapping requires exactly one canonical 'empty' category.")
    animals = sorted((item for item in normalized if item[1] != "empty"), key=lambda item: item[1])
    ordered = [empty[0], *animals]
    mapping = {original_id: internal_id for internal_id, (original_id, _) in enumerate(ordered)}
    names = tuple(name for _, name in ordered)
    checksum = hashlib.sha256("\n".join(f"{index}:{name}" for index, name in enumerate(names)).encode("utf-8")).hexdigest()
    return mapping, names, checksum


def build_cct_tpa_logits(
    features: torch.Tensor,
    text_directions: torch.Tensor,
    prototypes: torch.Tensor,
    present_mask: torch.Tensor,
    logit_scale: torch.Tensor | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    normalized_features = F.normalize(features.float(), dim=1)
    normalized_text = F.normalize(text_directions.float(), dim=1)
    normalized_prototypes = F.normalize(prototypes.float(), dim=1)
    scale = float(torch.as_tensor(logit_scale).exp().item())
    text_logits = scale * normalized_features @ normalized_text.t()
    prototype_residual = CCT_PROTOTYPE_SCALE * normalized_features @ normalized_prototypes.t()
    if not present_mask.all():
        prototype_residual[:, ~present_mask] = 0.0
    combined = text_logits + prototype_residual
    diagnostics = {
        "text_logit_std": float(text_logits.std().item()),
        "prototype_logit_std": float(prototype_residual.std().item()),
        "combined_logit_std": float(combined.std().item()),
        "text_logit_scale": scale,
        "prototype_scale": CCT_PROTOTYPE_SCALE,
    }
    return combined, diagnostics


def validate_cct_split_locations(split_locations: Mapping[str, Sequence[str]]) -> dict:
    trans_validation = set(str(value) for value in split_locations.get("trans_validation", ()))
    trans_test = set(str(value) for value in split_locations.get("trans_test", ()))
    if len(trans_validation) != CCT_EXPECTED_TRANS_VALIDATION_LOCATIONS:
        raise ValueError("CCT-20 trans-validation must contain exactly one location.")
    if len(trans_test) != CCT_EXPECTED_TRANS_TEST_LOCATIONS:
        raise ValueError("CCT-20 trans-test must contain exactly nine locations.")
    if trans_validation & trans_test:
        raise ValueError("CCT-20 trans-validation and trans-test locations must be disjoint.")
    return {
        "trans_validation_location_count": len(trans_validation),
        "trans_test_location_count": len(trans_test),
        "limited_cluster_uncertainty": True,
    }


def verify_cct_stage(stage: str, preregistration_path: Path, ledger_path: Path) -> dict:
    if stage not in {"cis_validation", "trans_validation", "trans_test"}:
        raise ValueError(f"Unsupported CCT evaluation stage: {stage!r}")
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    required = {
        "vfep_eta_grid",
        "stp_eta_grid",
        "control_eta_policy",
        "primary_metric_labels",
        "tie_tolerance",
        "tie_break_rule",
        "dataset_snapshot_sha256",
        "source_bundle_sha256",
    }
    missing = sorted(required - preregistration.keys())
    if missing:
        raise ValueError(f"CCT preregistration is incomplete: {missing}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    completed = ledger.get("completed_stages", [])
    if stage in completed:
        raise ValueError(f"CCT {stage} has already produced a successful receipt.")
    if stage == "trans_test" and not preregistration.get("trans_validation_preflight_passed", False):
        raise ValueError("CCT trans-test is locked until trans-validation preflight passes.")
    return preregistration


def _atomic_write_json(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def write_cct_receipt(
    stage: str,
    output_path: Path,
    preregistration_path: Path,
    report_path: Path,
    ledger_path: Path,
) -> None:
    verify_cct_stage(stage, preregistration_path, ledger_path)
    report_checksum = hashlib.sha256(report_path.read_bytes()).hexdigest()
    preregistration_checksum = hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
    payload = {
        "stage": stage,
        "preregistration_sha256": preregistration_checksum,
        "report_sha256": report_checksum,
    }
    _atomic_write_json(output_path, payload)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    completed = list(ledger.get("completed_stages", []))
    if stage in completed:
        raise ValueError(f"CCT {stage} has already produced a successful receipt.")
    completed.append(stage)
    receipts = list(ledger.get("receipts", []))
    receipts.append({
        "stage": stage,
        "receipt_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    })
    _atomic_write_json(ledger_path, {**ledger, "completed_stages": completed, "receipts": receipts})
