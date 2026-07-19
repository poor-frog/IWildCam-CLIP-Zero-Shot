from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from src.models.lesvi import (
    LesviPrior,
    DonorRotation,
    build_donor_event_rotation,
    build_lesvi_logits,
    estimate_lesvi_prior,
    with_visibility_variant,
)
from src.models.lesvi_evaluation import MetricSupport, build_confirmation_report, build_metric_support
from src.models.stmp_adapter import apply_sequence_consensus, metadata_group_key
from src.models.vfep_cct20 import canonical_cct_mapping


CCT_SEQUENCE_FIELD_INDEX = 1
CCT_LOCATION_FIELD_INDEX = 0
CCT_ROTATION_SEEDS = tuple(range(20260718, 20260817))


@dataclass(frozen=True, slots=True)
class CctRecord:
    image_id: str
    image_path: Path
    label: int
    location: str
    sequence: str | None


@dataclass(frozen=True, slots=True)
class CctEvaluationSupport:
    metric_support: MetricSupport
    context_eligible_mask: torch.Tensor
    common_rotation_mask: torch.Tensor
    rotations: tuple[DonorRotation, ...]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cct_snapshot_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(value).resolve() for value in paths), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_cct_metadata_manifest(annotation_path: Path) -> dict[str, object]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = payload.get("categories")
    images = payload.get("images")
    if not isinstance(categories, list) or not isinstance(images, list):
        raise ValueError("CCT metadata manifest requires categories and images lists.")
    _, class_names, mapping_sha256 = canonical_cct_mapping(categories)
    locations = sorted({
        _required_metadata_value(image, ("location", "location_id", "camera", "camera_id"), "location")
        for image in images
        if isinstance(image, dict)
    })
    if not locations or not images:
        raise ValueError("CCT metadata manifest must contain images with valid locations.")
    return {
        "image_count": len(images),
        "locations": locations,
        "class_names": list(class_names),
        "class_mapping_sha256": mapping_sha256,
    }


def cct_label_diagnostics(annotation_path: Path) -> dict[str, int]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("CCT label diagnostics require images and annotations lists.")
    labels_by_image: dict[str, set[int]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("CCT annotations must be JSON objects.")
        labels_by_image.setdefault(str(annotation.get("image_id")), set()).add(int(annotation["category_id"]))
    single = sum(len(labels_by_image.get(str(image.get("id")), set())) == 1 for image in images)
    multi = sum(len(labels_by_image.get(str(image.get("id")), set())) > 1 for image in images)
    return {
        "image_count": len(images),
        "single_label_image_count": single,
        "excluded_multi_label_image_count": multi,
        "unlabeled_image_count": len(images) - single - multi,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify_lesvi_cct_stage(stage: str, frozen_spec_path: Path, ledger_path: Path) -> dict[str, object]:
    if stage not in {"cis_validation", "trans_validation", "trans_test"}:
        raise ValueError(f"Unsupported CCT LESVI stage: {stage!r}.")
    spec = json.loads(frozen_spec_path.read_text(encoding="utf-8"))
    required = {
        "method",
        "dataset_snapshot_sha256",
        "source_bundle_sha256",
        "class_mapping_sha256",
        "prior_artifact_sha256",
    }
    if spec.get("method") != "LESVI-CCT20-v0" or required - spec.keys():
        raise ValueError("CCT LESVI frozen specification is incomplete or has the wrong method.")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if ledger.get("method") != "LESVI-CCT20-v0":
        raise ValueError("CCT LESVI ledger has the wrong method.")
    if ledger.get("frozen_spec_sha256") != file_sha256(frozen_spec_path):
        raise ValueError("CCT LESVI ledger does not match the frozen specification.")
    completed = ledger.get("completed_stages", [])
    if not isinstance(completed, list):
        raise ValueError("CCT LESVI ledger must contain completed_stages.")
    if stage in completed:
        raise ValueError(f"CCT LESVI {stage} already has a successful receipt.")
    if stage == "trans_validation" and "cis_validation" not in completed:
        raise ValueError("CCT LESVI trans-validation is locked until cis-validation completes.")
    if stage == "trans_test":
        receipts = ledger.get("receipts", [])
        preflight = any(
            isinstance(receipt, dict)
            and receipt.get("stage") == "trans_validation"
            and receipt.get("directional_preflight_passed") is True
            for receipt in receipts
        )
        if not preflight:
            raise ValueError("CCT LESVI trans-test is locked until trans-validation directional preflight passes.")
    return spec


def write_lesvi_cct_receipt(
    stage: str,
    report_path: Path,
    frozen_spec_path: Path,
    ledger_path: Path,
    receipt_path: Path,
) -> None:
    verify_lesvi_cct_stage(stage, frozen_spec_path, ledger_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    directional = report.get("cct20", {}).get("directional_preflight_passed") if isinstance(report.get("cct20"), dict) else None
    receipt = {
        "stage": stage,
        "frozen_spec_sha256": file_sha256(frozen_spec_path),
        "report_sha256": file_sha256(report_path),
        "directional_preflight_passed": directional,
    }
    _atomic_write_json(receipt_path, receipt)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    completed = list(ledger.get("completed_stages", []))
    completed.append(stage)
    receipts = list(ledger.get("receipts", []))
    receipts.append(receipt)
    _atomic_write_json(ledger_path, {
        "method": ledger["method"],
        "frozen_spec_sha256": ledger["frozen_spec_sha256"],
        "completed_stages": completed,
        "receipts": receipts,
    })


def _required_metadata_value(image: Mapping[str, object], keys: Sequence[str], label: str) -> str:
    for key in keys:
        value = image.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"CCT image {image.get('id')!r} is missing {label} metadata.")


def load_cct_records(annotation_path: Path, image_root: Path) -> tuple[tuple[CctRecord, ...], tuple[str, ...], str]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = payload.get("categories")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(categories, list) or not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("CCT annotation file must contain categories, images, and annotations lists.")
    category_mapping, class_names, mapping_sha256 = canonical_cct_mapping(categories)
    labels_by_image: dict[str, set[int]] = {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError("CCT annotations must be JSON objects.")
        image_id = str(annotation.get("image_id"))
        original_category = int(annotation["category_id"])
        if original_category not in category_mapping:
            raise ValueError(f"CCT annotation uses unknown category {original_category}.")
        labels_by_image.setdefault(image_id, set()).add(category_mapping[original_category])

    records: list[CctRecord] = []
    for image in images:
        if not isinstance(image, dict):
            raise ValueError("CCT images must be JSON objects.")
        image_id = str(image.get("id"))
        labels = labels_by_image.get(image_id, set())
        if len(labels) > 1:
            continue
        if len(labels) == 0:
            raise ValueError(f"CCT image {image_id!r} must have exactly one image-level label.")
        location = _required_metadata_value(image, ("location", "location_id", "camera", "camera_id"), "location")
        raw_sequence = next(
            (str(image[key]).strip() for key in ("seq_id", "sequence_id", "sequence") if image.get(key) is not None and str(image[key]).strip()),
            None,
        )
        sequence = None if raw_sequence is None else f"{location}::{raw_sequence}"
        file_name = _required_metadata_value(image, ("file_name",), "file name")
        records.append(
            CctRecord(
                image_id=image_id,
                image_path=(image_root / file_name).resolve(),
                label=next(iter(labels)),
                location=location,
                sequence=sequence,
            )
        )
    records.sort(key=lambda record: record.image_id)
    return tuple(records), class_names, mapping_sha256


def records_metadata(records: Sequence[CctRecord]) -> tuple[tuple[str, str | None], ...]:
    return tuple((record.location, record.sequence) for record in records)


def estimate_cct_lesvi_prior(records: Sequence[CctRecord], class_count: int, class_mapping_sha256: str) -> LesviPrior:
    labels = torch.tensor([record.label for record in records], dtype=torch.long)
    return estimate_lesvi_prior(
        labels,
        records_metadata(records),
        sequence_field_index=CCT_SEQUENCE_FIELD_INDEX,
        location_field_index=CCT_LOCATION_FIELD_INDEX,
        class_count=class_count,
        class_mapping_sha256=class_mapping_sha256,
    )


def _context_eligible(metadata: Sequence) -> torch.Tensor:
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(metadata):
        sequence = metadata_group_key(row, CCT_SEQUENCE_FIELD_INDEX)
        if sequence is not None:
            grouped.setdefault(sequence, []).append(index)
    result = torch.zeros(len(metadata), dtype=torch.bool)
    for indices in grouped.values():
        if len(indices) >= 2:
            result[torch.tensor(indices, dtype=torch.long)] = True
    return result


def build_cct_evaluation_support(
    records: Sequence[CctRecord],
    labels: torch.Tensor,
    rotation_seeds: Sequence[int] = CCT_ROTATION_SEEDS,
) -> CctEvaluationSupport:
    if labels.shape[0] != len(records):
        raise ValueError("CCT labels and records must align.")
    if len(rotation_seeds) != 99:
        raise ValueError("CCT LESVI confirmation requires exactly 99 donor rotations.")
    metadata = records_metadata(records)
    context_eligible = _context_eligible(metadata)
    rotations = tuple(
        build_donor_event_rotation(
            metadata,
            sequence_field_index=CCT_SEQUENCE_FIELD_INDEX,
            location_field_index=CCT_LOCATION_FIELD_INDEX,
            seed=int(seed),
        )
        for seed in rotation_seeds
    )
    common_rotation = context_eligible.clone()
    for rotation in rotations:
        common_rotation &= rotation.available_mask
    metric_support = build_metric_support(
        labels,
        metadata,
        sequence_field_index=CCT_SEQUENCE_FIELD_INDEX,
        rotation_common_mask=common_rotation,
    )
    return CctEvaluationSupport(metric_support, context_eligible, common_rotation, rotations)


def evaluate_cct_lesvi(
    *,
    stage: str,
    labels: torch.Tensor,
    tpa_logits: torch.Tensor,
    records: Sequence[CctRecord],
    prior: LesviPrior,
    rotation_seeds: Sequence[int] = CCT_ROTATION_SEEDS,
    frozen_support: CctEvaluationSupport | None = None,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    if stage not in {"cis_validation", "trans_validation", "trans_test"}:
        raise ValueError(f"Unsupported CCT LESVI stage: {stage!r}.")
    if labels.shape[0] != len(records) or tpa_logits.shape[0] != len(records):
        raise ValueError("CCT labels, logits, and records must align.")
    metadata = records_metadata(records)
    locations = [record.location for record in records]
    evaluation_support = frozen_support or build_cct_evaluation_support(records, labels, rotation_seeds)
    context_eligible = evaluation_support.context_eligible_mask
    rotations = evaluation_support.rotations
    common_rotation = evaluation_support.common_rotation_mask

    lesvi = build_lesvi_logits(tpa_logits, metadata, sequence_field_index=CCT_SEQUENCE_FIELD_INDEX, prior=prior)
    no_visibility = build_lesvi_logits(
        tpa_logits,
        metadata,
        sequence_field_index=CCT_SEQUENCE_FIELD_INDEX,
        prior=with_visibility_variant(prior, "none"),
    )
    stp = apply_sequence_consensus(tpa_logits, metadata, CCT_SEQUENCE_FIELD_INDEX, 0.5)
    rotated = tuple(
        build_lesvi_logits(
            tpa_logits,
            metadata,
            sequence_field_index=CCT_SEQUENCE_FIELD_INDEX,
            prior=prior,
            support_by_target=rotation.support_by_target,
        ).logits
        for rotation in rotations
    )
    support = evaluation_support.metric_support
    logits_by_name = {
        "tpa": tpa_logits,
        "stp_mean": stp,
        "lesvi": lesvi.logits,
        "no_visibility": no_visibility.logits,
    }
    report = build_confirmation_report(
        labels=labels,
        logits_by_name=logits_by_name,
        support=support,
        location_keys=locations,
        rotation_logits=rotated,
        rotation_common_mask=common_rotation,
        context_eligible_mask=context_eligible,
        rotation_fallback_counts=[rotation.fallback_assignment_count for rotation in rotations],
    )
    stp_overall = float(report["methods"]["stp_mean"]["overall"]["macro_f1"])
    lesvi_overall = float(report["methods"]["lesvi"]["overall"]["macro_f1"])
    stp_animal = float(report["methods"]["stp_mean"]["animal_true_nonempty"]["macro_f1"])
    lesvi_animal = float(report["methods"]["lesvi"]["animal_true_nonempty"]["macro_f1"])
    directional_preflight = lesvi_overall > stp_overall and lesvi_animal >= stp_animal
    report["cct20"] = {
        "stage": stage,
        "directional_preflight_passed": directional_preflight if stage == "trans_validation" else None,
        "trans_test_opened": stage == "trans_test",
        "limited_cluster_uncertainty": stage == "trans_test",
    }
    return report, logits_by_name
