import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.lesvi_candidate_preflight import (
    FROZEN_GATES,
    canonical_json_bytes,
    canonical_json_sha256,
    run_candidate_preflight,
)


SHA = "0" * 64


def _record(frame, event, location, label):
    return {
        "frame_id": frame,
        "event_id": event,
        "location_id": location,
        "labels": [label],
    }


def _metadata(dataset_id="viable", *, failure=None):
    classes = [{"id": index, "name": "empty" if index == 0 else f"species-{index}"} for index in range(7)]
    train = [
        _record("empty-0", "empty", "train-0", 0),
        _record("empty-1", "empty", "train-0", 0),
        _record("mixed-0", "mixed", "train-1", 1),
        _record("mixed-1", "mixed", "train-1", 0),
    ]
    train.extend(_record(f"species-{label}", f"event-{label}", f"train-{label}", label) for label in range(2, 7))
    confirmation = [
        _record(f"confirm-{index}", f"confirm-event-{index}", f"location-{index}", index % 7)
        for index in range(14)
    ]
    if failure == "cct20_prior":
        train = [_record("animal", "animal", "train", 1)]
    elif failure == "iwildcam_coverage":
        confirmation = [
            _record(f"confirm-{index}", f"confirm-event-{index}", f"location-{index % 12}", index % 4)
            for index in range(24)
        ]
    return {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "snapshot_id": "snapshot-v1",
        "empty_class_id": 0,
        "classes": classes,
        "annotation_sources": {
            "train": {"uri": "train.json", "sha256": SHA},
            "confirmation": {"uri": "confirmation.json", "sha256": SHA},
            "test": {"uri": "test.json", "sha256": SHA},
        },
        "access_policy": {"confirmation": "aggregate_only", "test_labels": "unopened"},
        "splits": {
            "train": {"records": train},
            "confirmation": {"records": confirmation},
            "test": {"record_count": 100, "location_count": 20},
        },
    }


def _write_registry(tmp_path, candidates):
    entries = []
    for dataset_id, metadata in candidates:
        metadata_path = tmp_path / f"{dataset_id}.json"
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        entries.append({
            "dataset_id": dataset_id,
            "snapshot_id": metadata["snapshot_id"],
            "metadata_path": metadata_path.name,
            "metadata_canonical_sha256": canonical_json_sha256(metadata),
        })
    registry = {"schema_version": 1, "gates": FROZEN_GATES, "candidates": entries}
    registry_path = tmp_path / "registry.json"
    registry_path.write_bytes(canonical_json_bytes(registry))
    return registry_path


def test_metadata_preflight_selects_viable_candidate_and_writes_aggregate_only_receipt(tmp_path):
    registry = _write_registry(tmp_path, [("viable", _metadata())])
    output = tmp_path / "output"

    selection = run_candidate_preflight(registry, output)

    receipt = json.loads((output / "candidates/viable/preflight_receipt.json").read_text(encoding="utf-8"))
    assert selection["status"] == "selected"
    assert selection["selected_dataset_id"] == "viable"
    assert receipt["viability_pass"] is True
    assert receipt["train"]["valid_all_empty_event_count"] == 1
    assert receipt["train"]["mixed_visibility_species_event_count"] == 1
    assert receipt["confirmation"]["supported_tail_class_count"] == 6
    assert receipt["access"] == {
        "confirmation_annotations": "aggregate_only",
        "confirmation_predictions_opened": False,
        "test_labels_opened": False,
        "model_or_images_loaded": False,
    }
    assert "labels" not in json.dumps(receipt["confirmation"])
    assert (output / "registry_lock.json").read_bytes().endswith(b"\n")


def test_iwildcam_style_coverage_failure_and_cct20_prior_failure_remain_blocked(tmp_path):
    registry = _write_registry(tmp_path, [
        ("iwildcam", _metadata("iwildcam", failure="iwildcam_coverage")),
        ("cct20", _metadata("cct20", failure="cct20_prior")),
    ])
    output = tmp_path / "output"

    selection = run_candidate_preflight(registry, output)

    iwildcam = json.loads((output / "candidates/iwildcam/preflight_receipt.json").read_text(encoding="utf-8"))
    cct20 = json.loads((output / "candidates/cct20/preflight_receipt.json").read_text(encoding="utf-8"))
    assert selection["status"] == "closed"
    assert selection["reason"] == "confirmation_unavailable_on_screened_benchmarks"
    assert iwildcam["checks"]["confirmation_supported_class_fraction"] is False
    assert iwildcam["checks"]["confirmation_supported_tail_class_count"] is False
    assert cct20["checks"]["has_train_all_empty_event"] is False
    assert cct20["checks"]["has_train_species_event_with_empty_frame"] is False
    assert selection["test_labels_opened"] is False


def test_selection_is_deterministic_and_uses_dataset_id_as_final_tie_break(tmp_path):
    registry = _write_registry(tmp_path, [
        ("zeta", _metadata("zeta")),
        ("alpha", _metadata("alpha")),
    ])

    first = run_candidate_preflight(registry, tmp_path / "first")
    second = run_candidate_preflight(registry, tmp_path / "second")

    assert first == second
    assert first["passing_candidate_ids_ranked"] == ["alpha", "zeta"]
    assert first["selected_dataset_id"] == "alpha"
    assert (tmp_path / "first/selection_receipt.json").read_bytes() == (
        tmp_path / "second/selection_receipt.json"
    ).read_bytes()


def test_registry_rejects_changed_gates_duplicate_ids_and_metadata_checksum(tmp_path):
    registry = _write_registry(tmp_path, [("candidate", _metadata("candidate"))])
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["gates"]["max_confirmation_largest_location_frame_fraction"] = 0.25
    registry.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="frozen LESVI-v0 gates"):
        run_candidate_preflight(registry, tmp_path / "changed-gate")

    duplicate = _write_registry(tmp_path, [
        ("duplicate", _metadata("duplicate")),
        ("duplicate", _metadata("duplicate")),
    ])
    with pytest.raises(ValueError, match="duplicate dataset_id"):
        run_candidate_preflight(duplicate, tmp_path / "duplicate-output")

    checksum = _write_registry(tmp_path, [("checksum", _metadata("checksum"))])
    checksum_payload = json.loads(checksum.read_text(encoding="utf-8"))
    checksum_payload["candidates"][0]["metadata_canonical_sha256"] = "f" * 64
    checksum.write_bytes(canonical_json_bytes(checksum_payload))
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        run_candidate_preflight(checksum, tmp_path / "checksum-output")


def test_receipts_are_immutable_and_metadata_path_cannot_escape_registry(tmp_path):
    registry = _write_registry(tmp_path, [("candidate", _metadata("candidate"))])
    output = tmp_path / "output"
    run_candidate_preflight(registry, output)

    with pytest.raises(ValueError, match="receipts are immutable"):
        run_candidate_preflight(registry, output)

    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["candidates"][0]["metadata_path"] = "../candidate.json"
    registry.write_bytes(canonical_json_bytes(payload))
    with pytest.raises(ValueError, match="inside the registry directory"):
        run_candidate_preflight(registry, tmp_path / "escaped")


def test_test_split_rejects_records_or_labels(tmp_path):
    metadata = _metadata("leaky")
    metadata["splits"]["test"]["records"] = [_record("test", "test", "test", 1)]
    registry = _write_registry(tmp_path, [("leaky", metadata)])

    with pytest.raises(ValueError, match="Test metadata may contain only"):
        run_candidate_preflight(registry, tmp_path / "output")


def test_cli_runs_without_inference_dependencies_and_creates_no_frozen_spec(tmp_path):
    registry = _write_registry(tmp_path, [("blocked", _metadata("blocked", failure="cct20_prior"))])
    output = tmp_path / "cli-output"

    result = subprocess.run(
        [
            sys.executable,
            "src/prepare_lesvi_candidate_preflight.py",
            "--registry",
            str(registry),
            "--output-dir",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "LESVI candidate preflight closed" in result.stdout
    assert not list(output.rglob("*frozen_spec*"))
    assert not list(output.rglob("*logits*"))
    assert not list(output.rglob("*checkpoint*"))
