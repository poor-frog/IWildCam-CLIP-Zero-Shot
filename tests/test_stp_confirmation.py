import json
from pathlib import Path

import pytest


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_confirmation_guard_rejects_missing_spec_before_any_execution(tmp_path):
    from src.models.stp_confirmation import StpConfirmationError, verify_confirmation_ready

    ledger = tmp_path / "ledger.json"
    _write(ledger, json.dumps({"confirmations": []}))

    with pytest.raises(StpConfirmationError, match="Frozen specification"):
        verify_confirmation_ready(tmp_path / "missing.json", ledger, tmp_path)


def test_source_bundle_checksum_detects_prediction_path_change(tmp_path):
    from src.models.stp_confirmation import build_source_bundle_checksum

    source = tmp_path / "module.py"
    _write(source, "value = 1\n")
    first = build_source_bundle_checksum(tmp_path, ("module.py",))
    _write(source, "value = 2\n")
    second = build_source_bundle_checksum(tmp_path, ("module.py",))

    assert first != second


def test_confirmation_guard_refuses_existing_matching_receipt(tmp_path):
    from src.models.stp_confirmation import StpConfirmationError, build_source_bundle_checksum, file_sha256, verify_confirmation_ready

    _write(tmp_path / "module.py", "value = 1\n")
    _write(tmp_path / "audit_manifest.json", "{}\n")
    _write(tmp_path / "class_mapping.json", "{}\n")
    ledger_path = tmp_path / "ledger.json"
    _write(ledger_path, json.dumps({"confirmations": [{"spec_sha256": "placeholder"}]}))
    source_checksum = build_source_bundle_checksum(tmp_path, ("module.py",))
    spec = {
        "confirmation_status": "frozen",
        "source_files": ["module.py"],
        "source_bundle_sha256": source_checksum,
        "audit_manifest_sha256": file_sha256(tmp_path / "audit_manifest.json"),
        "audit_manifest_path": "audit_manifest.json",
        "class_mapping_sha256": file_sha256(tmp_path / "class_mapping.json"),
        "class_mapping_path": "class_mapping.json",
        "confirmation_ledger_genesis_sha256": file_sha256(ledger_path),
    }
    spec_path = tmp_path / "spec.json"
    _write(spec_path, json.dumps(spec))

    ready = verify_confirmation_ready(spec_path, ledger_path, tmp_path)
    genesis_checksum = file_sha256(ledger_path)
    _write(ledger_path, json.dumps({"confirmations": [{"spec_sha256": ready.spec_sha256}], "genesis_ledger_sha256": genesis_checksum}))

    with pytest.raises(StpConfirmationError, match="already contains"):
        verify_confirmation_ready(spec_path, ledger_path, tmp_path)


def test_confirmation_guard_rejects_manifest_change_before_any_metric_work(tmp_path):
    from src.models.stp_confirmation import StpConfirmationError, build_source_bundle_checksum, file_sha256, verify_confirmation_ready

    _write(tmp_path / "module.py", "value = 1\n")
    _write(tmp_path / "audit_manifest.json", "{}\n")
    _write(tmp_path / "class_mapping.json", "{}\n")
    _write(tmp_path / "ledger.json", json.dumps({"confirmations": []}))
    spec = {
        "confirmation_status": "frozen",
        "source_files": ["module.py"],
        "source_bundle_sha256": build_source_bundle_checksum(tmp_path, ("module.py",)),
        "audit_manifest_path": "audit_manifest.json",
        "audit_manifest_sha256": file_sha256(tmp_path / "audit_manifest.json"),
        "class_mapping_path": "class_mapping.json",
        "class_mapping_sha256": file_sha256(tmp_path / "class_mapping.json"),
        "confirmation_ledger_genesis_sha256": file_sha256(tmp_path / "ledger.json"),
    }
    _write(tmp_path / "spec.json", json.dumps(spec))
    _write(tmp_path / "audit_manifest.json", '{"changed": true}\n')

    with pytest.raises(StpConfirmationError, match="audit manifest checksum"):
        verify_confirmation_ready(tmp_path / "spec.json", tmp_path / "ledger.json", tmp_path)
