from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA_VERSION = 1
FROZEN_GATES = {
    "required_all_empty_train_events": 1,
    "required_mixed_visibility_species_events": 1,
    "min_confirmation_supported_class_fraction": 0.8,
    "min_confirmation_supported_tail_class_count": 5,
    "min_confirmation_location_count": 10,
    "max_confirmation_largest_location_frame_fraction": 0.5,
    "tail_max_train_frames": 20,
}
_DATASET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def canonical_json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_json_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(payload))
    os.replace(temporary, path)


def _required_mapping(payload: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a JSON object.")
    return value


def _required_list(payload: Mapping[str, object], key: str) -> list[object]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON list.")
    return value


def _validate_sha256(value: object, label: str) -> str:
    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return text


def load_registry(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Candidate registry must be a JSON object.")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Candidate registry schema_version must be {SCHEMA_VERSION}.")
    if payload.get("gates") != FROZEN_GATES:
        raise ValueError("Candidate registry gates do not match the frozen LESVI-v0 gates.")
    candidates = _required_list(payload, "candidates")
    if not candidates:
        raise ValueError("Candidate registry must contain at least one candidate.")
    seen: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"Candidate {index} must be a JSON object.")
        dataset_id = str(candidate.get("dataset_id", ""))
        if _DATASET_ID.fullmatch(dataset_id) is None:
            raise ValueError(f"Candidate {index} has an invalid dataset_id.")
        if dataset_id in seen:
            raise ValueError(f"Candidate registry contains duplicate dataset_id {dataset_id!r}.")
        seen.add(dataset_id)
        if not str(candidate.get("snapshot_id", "")).strip():
            raise ValueError(f"Candidate {dataset_id!r} must include snapshot_id.")
        if not str(candidate.get("metadata_path", "")).strip():
            raise ValueError(f"Candidate {dataset_id!r} must include metadata_path.")
        _validate_sha256(candidate.get("metadata_canonical_sha256"), f"Candidate {dataset_id!r} metadata hash")
    return payload


def _validate_classes(metadata: Mapping[str, object]) -> tuple[int, tuple[int, ...]]:
    classes = _required_list(metadata, "classes")
    if not classes:
        raise ValueError("Candidate metadata must contain classes.")
    class_ids: list[int] = []
    for entry in classes:
        if not isinstance(entry, dict) or "id" not in entry or not str(entry.get("name", "")).strip():
            raise ValueError("Each class must contain an integer id and non-empty name.")
        class_ids.append(int(entry["id"]))
    if class_ids != list(range(len(class_ids))):
        raise ValueError("Candidate class ids must be ordered and contiguous from zero.")
    empty_class_id = int(metadata.get("empty_class_id", -1))
    if empty_class_id != 0:
        raise ValueError("LESVI candidate metadata must map empty to class id zero.")
    return empty_class_id, tuple(class_ids)


def _validate_sources(metadata: Mapping[str, object]) -> Mapping[str, object]:
    sources = _required_mapping(metadata, "annotation_sources")
    if set(sources) != {"train", "confirmation", "test"}:
        raise ValueError("annotation_sources must contain exactly train, confirmation, and test.")
    for split, source in sources.items():
        if not isinstance(source, dict) or not str(source.get("uri", "")).strip():
            raise ValueError(f"Annotation source {split!r} must contain a non-empty uri.")
        _validate_sha256(source.get("sha256"), f"Annotation source {split!r}")
    return sources


def _parse_records(
    records: Sequence[object],
    class_ids: set[int],
) -> tuple[list[tuple[str, str, int]], dict[str, int], set[str]]:
    valid: list[tuple[str, str, int]] = []
    counts = {
        "record_count": len(records),
        "valid_single_label_record_count": 0,
        "excluded_multi_label_record_count": 0,
        "excluded_unlabeled_record_count": 0,
        "malformed_record_count": 0,
    }
    invalid_events: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            counts["malformed_record_count"] += 1
            continue
        event_id = str(record.get("event_id", "")).strip()
        location_id = str(record.get("location_id", "")).strip()
        frame_id = str(record.get("frame_id", "")).strip()
        labels = record.get("labels")
        if not event_id or not location_id or not frame_id or not isinstance(labels, list):
            counts["malformed_record_count"] += 1
            if event_id:
                invalid_events.add(event_id)
            continue
        unique_labels: set[int] = set()
        try:
            unique_labels = {int(label) for label in labels}
        except (TypeError, ValueError):
            counts["malformed_record_count"] += 1
            invalid_events.add(event_id)
            continue
        if not unique_labels:
            counts["excluded_unlabeled_record_count"] += 1
            invalid_events.add(event_id)
            continue
        if not unique_labels <= class_ids:
            counts["malformed_record_count"] += 1
            invalid_events.add(event_id)
            continue
        if len(unique_labels) > 1:
            counts["excluded_multi_label_record_count"] += 1
            invalid_events.add(event_id)
            continue
        valid.append((event_id, location_id, next(iter(unique_labels))))
        counts["valid_single_label_record_count"] += 1
    return valid, counts, invalid_events


def _train_aggregates(
    records: Sequence[object],
    class_ids: tuple[int, ...],
    empty_class_id: int,
) -> dict[str, object]:
    valid, counts, invalid_events = _parse_records(records, set(class_ids))
    grouped: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for event_id, location_id, label in valid:
        grouped[event_id].append((location_id, label))
    label_counts: Counter[int] = Counter()
    valid_empty_events = 0
    valid_species_events = 0
    mixed_visibility_species_events = 0
    excluded_multi_species_events = 0
    excluded_inconsistent_location_events = 0
    for event_id, rows in grouped.items():
        if event_id in invalid_events:
            continue
        locations = {location for location, _ in rows}
        if len(locations) != 1:
            excluded_inconsistent_location_events += 1
            continue
        labels = [label for _, label in rows]
        species = {label for label in labels if label != empty_class_id}
        if not species:
            valid_empty_events += 1
            label_counts.update(labels)
        elif len(species) == 1:
            valid_species_events += 1
            if empty_class_id in labels:
                mixed_visibility_species_events += 1
            label_counts.update(labels)
        else:
            excluded_multi_species_events += 1
    tail_max = int(FROZEN_GATES["tail_max_train_frames"])
    tail_classes = {
        class_id
        for class_id in class_ids
        if class_id != empty_class_id and 0 < label_counts[class_id] <= tail_max
    }
    return {
        **counts,
        "invalid_event_count": len(invalid_events),
        "valid_all_empty_event_count": valid_empty_events,
        "valid_single_species_event_count": valid_species_events,
        "mixed_visibility_species_event_count": mixed_visibility_species_events,
        "excluded_multi_species_event_count": excluded_multi_species_events,
        "excluded_inconsistent_location_event_count": excluded_inconsistent_location_events,
        "tail_class_ids": tail_classes,
    }


def _confirmation_aggregates(
    records: Sequence[object],
    class_ids: tuple[int, ...],
    tail_classes: set[int],
) -> dict[str, object]:
    valid, counts, _ = _parse_records(records, set(class_ids))
    label_counts = Counter(label for _, _, label in valid)
    location_counts = Counter(location for _, location, _ in valid)
    supported_class_count = sum(label_counts[class_id] > 0 for class_id in class_ids)
    supported_tail_count = sum(label_counts[class_id] > 0 for class_id in tail_classes)
    valid_count = len(valid)
    largest_fraction = max(location_counts.values(), default=0) / valid_count if valid_count else 1.0
    return {
        **counts,
        "supported_class_count": supported_class_count,
        "supported_class_fraction": supported_class_count / len(class_ids),
        "supported_tail_class_count": supported_tail_count,
        "location_count": len(location_counts),
        "largest_location_frame_fraction": largest_fraction,
    }


def build_candidate_receipt(
    candidate: Mapping[str, object],
    metadata: Mapping[str, object],
    registry_canonical_sha256: str,
    metadata_raw_sha256: str,
) -> dict[str, object]:
    dataset_id = str(candidate["dataset_id"])
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Candidate {dataset_id!r} metadata schema_version must be {SCHEMA_VERSION}.")
    if metadata.get("dataset_id") != dataset_id or metadata.get("snapshot_id") != candidate.get("snapshot_id"):
        raise ValueError(f"Candidate {dataset_id!r} metadata identity does not match the registry.")
    access = _required_mapping(metadata, "access_policy")
    if access != {"confirmation": "aggregate_only", "test_labels": "unopened"}:
        raise ValueError(f"Candidate {dataset_id!r} has an invalid label-access policy.")
    empty_class_id, class_ids = _validate_classes(metadata)
    sources = _validate_sources(metadata)
    splits = _required_mapping(metadata, "splits")
    if set(splits) != {"train", "confirmation", "test"}:
        raise ValueError("splits must contain exactly train, confirmation, and test.")
    train = _required_mapping(splits, "train")
    confirmation = _required_mapping(splits, "confirmation")
    test = _required_mapping(splits, "test")
    if set(test) - {"record_count", "location_count"}:
        raise ValueError("Test metadata may contain only record_count and location_count; test labels must remain unopened.")
    if any(key in test for key in ("records", "labels", "class_counts")):
        raise ValueError("Test labels must not appear in candidate metadata.")
    train_aggregate = _train_aggregates(_required_list(train, "records"), class_ids, empty_class_id)
    tail_classes = train_aggregate.pop("tail_class_ids")
    confirmation_aggregate = _confirmation_aggregates(
        _required_list(confirmation, "records"), class_ids, tail_classes
    )
    checks = {
        "has_train_all_empty_event": train_aggregate["valid_all_empty_event_count"]
        >= FROZEN_GATES["required_all_empty_train_events"],
        "has_train_species_event_with_empty_frame": train_aggregate["mixed_visibility_species_event_count"]
        >= FROZEN_GATES["required_mixed_visibility_species_events"],
        "global_laplace_fallback_is_train_only": True,
        "has_event_and_location_metadata": train_aggregate["valid_single_label_record_count"] > 0
        and confirmation_aggregate["valid_single_label_record_count"] > 0,
        "confirmation_supported_class_fraction": confirmation_aggregate["supported_class_fraction"]
        >= FROZEN_GATES["min_confirmation_supported_class_fraction"],
        "confirmation_supported_tail_class_count": confirmation_aggregate["supported_tail_class_count"]
        >= FROZEN_GATES["min_confirmation_supported_tail_class_count"],
        "confirmation_location_count": confirmation_aggregate["location_count"]
        >= FROZEN_GATES["min_confirmation_location_count"],
        "confirmation_largest_location_fraction": confirmation_aggregate["largest_location_frame_fraction"]
        <= FROZEN_GATES["max_confirmation_largest_location_frame_fraction"],
    }
    viability_pass = all(checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "LESVI-v0",
        "dataset_id": dataset_id,
        "snapshot_id": str(candidate["snapshot_id"]),
        "status": "passed" if viability_pass else "blocked",
        "reason": None if viability_pass else "candidate_failed_metadata_preflight",
        "viability_pass": viability_pass,
        "gates": FROZEN_GATES,
        "checks": checks,
        "train": train_aggregate,
        "confirmation": confirmation_aggregate,
        "test_manifest": {
            "record_count": int(test.get("record_count", 0)),
            "location_count": int(test.get("location_count", 0)),
            "labels_opened": False,
        },
        "access": {
            "confirmation_annotations": "aggregate_only",
            "confirmation_predictions_opened": False,
            "test_labels_opened": False,
            "model_or_images_loaded": False,
        },
        "annotation_sources": sources,
        "registry_canonical_sha256": registry_canonical_sha256,
        "metadata_canonical_sha256": canonical_json_sha256(metadata),
        "metadata_raw_transport_sha256": metadata_raw_sha256,
    }


def select_candidate(receipts: Sequence[Mapping[str, object]], registry_sha256: str) -> dict[str, object]:
    passing = [receipt for receipt in receipts if receipt.get("viability_pass") is True]
    ranked = sorted(
        passing,
        key=lambda receipt: (
            -float(_required_mapping(receipt, "confirmation")["supported_class_fraction"]),
            -int(_required_mapping(receipt, "confirmation")["supported_tail_class_count"]),
            -int(_required_mapping(receipt, "confirmation")["location_count"]),
            str(receipt["dataset_id"]),
        ),
    )
    selected = str(ranked[0]["dataset_id"]) if ranked else None
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "LESVI-v0",
        "status": "selected" if selected is not None else "closed",
        "reason": None if selected is not None else "confirmation_unavailable_on_screened_benchmarks",
        "registry_canonical_sha256": registry_sha256,
        "ordered_candidate_ids": [str(receipt["dataset_id"]) for receipt in receipts],
        "passing_candidate_ids_ranked": [str(receipt["dataset_id"]) for receipt in ranked],
        "selected_dataset_id": selected,
        "selection_uses_model_performance": False,
        "test_labels_opened": False,
    }


def run_candidate_preflight(registry_path: Path, output_dir: Path) -> dict[str, object]:
    registry_path = registry_path.resolve()
    registry = load_registry(registry_path)
    registry_sha = canonical_json_sha256(registry)
    receipts: list[dict[str, object]] = []
    for candidate in _required_list(registry, "candidates"):
        if not isinstance(candidate, dict):
            raise AssertionError("Registry validation must reject non-object candidates.")
        relative_metadata_path = Path(str(candidate["metadata_path"]))
        if relative_metadata_path.is_absolute():
            raise ValueError("Candidate metadata_path must be relative to the registry directory.")
        metadata_path = (registry_path.parent / relative_metadata_path).resolve()
        if not metadata_path.is_relative_to(registry_path.parent):
            raise ValueError("Candidate metadata_path must remain inside the registry directory.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError(f"Candidate {candidate['dataset_id']!r} metadata must be a JSON object.")
        expected = _validate_sha256(candidate["metadata_canonical_sha256"], "metadata_canonical_sha256")
        if canonical_json_sha256(metadata) != expected:
            raise ValueError(f"Candidate {candidate['dataset_id']!r} metadata checksum mismatch.")
        receipt = build_candidate_receipt(candidate, metadata, registry_sha, file_sha256(metadata_path))
        receipts.append(receipt)
    selection = select_candidate(receipts, registry_sha)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError("Candidate preflight output directory already exists; receipts are immutable.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_json(temporary_dir / "registry_lock.json", {
            "schema_version": SCHEMA_VERSION,
            "registry": registry,
            "registry_canonical_sha256": registry_sha,
            "registry_raw_transport_sha256": file_sha256(registry_path),
        })
        for receipt in receipts:
            receipt_path = (
                temporary_dir / "candidates" / str(receipt["dataset_id"]) / "preflight_receipt.json"
            )
            atomic_write_json(receipt_path, receipt)
        atomic_write_json(temporary_dir / "selection_receipt.json", selection)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return selection
