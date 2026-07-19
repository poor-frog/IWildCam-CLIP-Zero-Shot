import json

import pytest
import torch

from src.models.lesvi_cct20 import (
    CctRecord,
    cct_snapshot_sha256,
    estimate_cct_lesvi_prior,
    evaluate_cct_lesvi,
    load_cct_records,
    verify_lesvi_cct_stage,
    write_lesvi_cct_receipt,
)


def _spec_and_ledger(tmp_path):
    spec_path = tmp_path / "spec.json"
    ledger_path = tmp_path / "ledger.json"
    spec_path.write_text(json.dumps({
        "method": "LESVI-CCT20-v0",
        "dataset_snapshot_sha256": "dataset",
        "source_bundle_sha256": "source",
        "class_mapping_sha256": "mapping",
        "prior_artifact_sha256": "prior",
    }), encoding="utf-8")
    import hashlib

    ledger_path.write_text(json.dumps({
        "method": "LESVI-CCT20-v0",
        "frozen_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        "completed_stages": [],
        "receipts": [],
    }), encoding="utf-8")
    return spec_path, ledger_path


def _records():
    records = []
    labels = []
    for location in ("a", "b"):
        for sequence_index, label in ((1, 1), (2, 2)):
            for frame in range(3):
                records.append(CctRecord(f"{location}-{sequence_index}-{frame}", __import__("pathlib").Path("x.jpg"), label, location, f"{location}::{sequence_index}"))
                labels.append(label)
    return tuple(records), torch.tensor(labels)


def test_cct_parser_canonicalizes_empty_and_prefixes_sequences_by_location(tmp_path):
    annotation = tmp_path / "train.json"
    annotation.write_text(json.dumps({
        "categories": [{"id": 30, "name": "empty"}, {"id": 6, "name": "Bobcat"}],
        "images": [{"id": "i", "file_name": "a.jpg", "location": 7, "seq_id": "s"}],
        "annotations": [{"id": "a", "image_id": "i", "category_id": 6}],
    }), encoding="utf-8")

    records, names, checksum = load_cct_records(annotation, tmp_path)

    assert names == ("empty", "bobcat")
    assert records[0].label == 1
    assert records[0].sequence == "7::s"
    assert len(checksum) == 64
    assert len(cct_snapshot_sha256((annotation,))) == 64


def test_cct_parser_excludes_multilabel_images_without_selecting_a_label(tmp_path):
    annotation = tmp_path / "train.json"
    annotation.write_text(json.dumps({
        "categories": [{"id": 30, "name": "empty"}, {"id": 6, "name": "bobcat"}, {"id": 10, "name": "rabbit"}],
        "images": [
            {"id": "single", "file_name": "single.jpg", "location": 7, "seq_id": "s"},
            {"id": "multi", "file_name": "multi.jpg", "location": 7, "seq_id": "s"},
        ],
        "annotations": [
            {"id": "a", "image_id": "single", "category_id": 6},
            {"id": "b", "image_id": "multi", "category_id": 6},
            {"id": "c", "image_id": "multi", "category_id": 10},
        ],
    }), encoding="utf-8")

    records, _, _ = load_cct_records(annotation, tmp_path)

    assert [record.image_id for record in records] == ["single"]


def test_cct_lesvi_prior_and_cis_evaluator_use_fixed_stp_and_99_rotations():
    records, labels = _records()
    prior = estimate_cct_lesvi_prior(records, 3, "mapping")
    logits = torch.full((len(records), 3), -1.0)
    logits[:, 0] = -2.0
    logits[labels == 1, 1] = 3.0
    logits[labels == 2, 2] = 3.0

    report, outputs = evaluate_cct_lesvi(
        stage="cis_validation",
        labels=labels,
        tpa_logits=logits,
        records=records,
        prior=prior,
    )

    assert set(outputs) == {"tpa", "stp_mean", "lesvi", "no_visibility"}
    assert report["cct20"]["stage"] == "cis_validation"
    assert report["cct20"]["trans_test_opened"] is False
    assert len(report["rotation"]["rotation_deltas_vs_tpa"]) == 99


def test_cct_trans_test_firewall_reads_preflight_from_immutable_ledger(tmp_path):
    spec_path, ledger_path = _spec_and_ledger(tmp_path)
    report_path = tmp_path / "report.json"
    receipt_path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="directional preflight"):
        verify_lesvi_cct_stage("trans_test", spec_path, ledger_path)

    cis_report_path = tmp_path / "cis-report.json"
    cis_report_path.write_text(json.dumps({"cct20": {"directional_preflight_passed": None}}), encoding="utf-8")
    write_lesvi_cct_receipt("cis_validation", cis_report_path, spec_path, ledger_path, tmp_path / "cis-receipt.json")
    report_path.write_text(json.dumps({"cct20": {"directional_preflight_passed": True}}), encoding="utf-8")
    write_lesvi_cct_receipt("trans_validation", report_path, spec_path, ledger_path, receipt_path)
    verify_lesvi_cct_stage("trans_test", spec_path, ledger_path)

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["receipts"][1]["directional_preflight_passed"] is True


def test_cct_trans_validation_requires_completed_cis_stage(tmp_path):
    spec_path, ledger_path = _spec_and_ledger(tmp_path)

    with pytest.raises(ValueError, match="cis-validation"):
        verify_lesvi_cct_stage("trans_validation", spec_path, ledger_path)
