import json
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np

from src.prepare_lesvi_freeze import SOURCE_FILES, _validate_inputs, main


class _TrainSubset:
    metadata_fields = ["location", "sequence_id"]
    y_array = np.asarray([0, 0, 1, 0, 1, 2], dtype=np.int64)
    metadata_array = np.asarray([[1, 1], [1, 1], [1, 2], [1, 2], [1, 2], [2, 3]], dtype=np.int64)


class _Dataset:
    n_classes = 3

    def get_subset(self, name, transform=None):
        assert name == "train"
        assert transform is None
        return _TrainSubset()


def test_prepare_freeze_runs_train_prior_then_synthetic_then_writes_frozen_spec(tmp_path, monkeypatch):
    audit = tmp_path / "audit.json"
    mapping = tmp_path / "mapping.json"
    ledger = tmp_path / "ledger.json"
    viability = tmp_path / "viability.json"
    checkpoint = tmp_path / "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
    output = tmp_path / "output"
    audit.write_text('{"phase":"val_audit_only","confirmation_performance_materialized":false,"foundation":"flyp","split_seed":"20260716"}\n')
    viability.write_text(
        '{"viability_pass":true,"audit_location_count":10,"confirm_location_count":10,'
        '"audit_supported_tail_class_count":5,"confirm_supported_tail_class_count":5,'
        '"confirm_supported_class_fraction":0.8,"confirm_largest_location_frame_fraction":0.5}\n'
    )
    mapping.write_text('{"classnames":["empty","a","b"],"class_mapping_sha256":"c67f66f0001ec01c69136584323e661e86d6d00a49fb9e8ca8b6b30613485eb9","empty_class_index":0}\n')
    ledger.write_text('{"confirmations":[]}\n')
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(sys.modules, "wilds", SimpleNamespace(get_dataset=lambda **_: _Dataset()))
    main(SimpleNamespace(
        data_location=tmp_path,
        audit_manifest=audit,
        class_mapping=mapping,
        viability_report=viability,
        confirmation_genesis_ledger=ledger,
        checkpoint=checkpoint,
        output_dir=output,
        workspace_root=__import__("pathlib").Path.cwd(),
    ))
    prior = json.loads((output / "lesvi_prior.json").read_text())
    synthetic = json.loads((output / "synthetic_verification.json").read_text())
    spec = json.loads((output / "lesvi_frozen_spec.json").read_text())
    assert prior["method"] == "LESVI-v0"
    assert synthetic["passed"] is True
    assert spec["confirmation_status"] == "frozen"
    assert spec["val_audit_lesvi_executed"] is False
    assert len(spec["rotation_seeds"]) == 99
    assert "src/config.py" in spec["source_files"]


def test_freeze_rejects_wrong_foundation_or_failed_viability():
    classnames = ["empty", "a"]
    import hashlib

    mapping_sha = hashlib.sha256("0:empty\n1:a".encode("utf-8")).hexdigest()
    mapping = {"classnames": classnames, "class_mapping_sha256": mapping_sha, "empty_class_index": 0}
    ledger = {"confirmations": []}
    viable = {
        "viability_pass": True,
        "audit_location_count": 10,
        "confirm_location_count": 10,
        "audit_supported_tail_class_count": 5,
        "confirm_supported_tail_class_count": 5,
        "confirm_supported_class_fraction": 0.8,
        "confirm_largest_location_frame_fraction": 0.5,
    }
    audit = {
        "phase": "val_audit_only",
        "confirmation_performance_materialized": False,
        "foundation": "drm_wise",
        "split_seed": "20260716",
    }
    import pytest

    with pytest.raises(ValueError, match="clean FLYP"):
        _validate_inputs(audit, mapping, ledger, viable)
    audit["foundation"] = "flyp"
    with pytest.raises(ValueError, match="failed viability"):
        _validate_inputs(audit, mapping, ledger, {"viability_pass": False})


def test_source_bundle_includes_runtime_configuration():
    assert "src/config.py" in SOURCE_FILES


def test_kaggle_freeze_publishes_blocked_receipt_without_opening_confirmation(tmp_path):
    module_path = __import__("pathlib").Path("kaggle-flyp-lesvi-freeze/kaggle_main.py")
    spec = importlib.util.spec_from_file_location("lesvi_freeze_launcher", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BLOCKED_OUTPUT_DIR = tmp_path / "blocked"
    audit = tmp_path / "audit.json"
    viability = tmp_path / "viability.json"
    audit.write_text('{"phase":"val_audit_only"}', encoding="utf-8")
    viability.write_text('{"viability_pass":false,"confirm_supported_class_fraction":0.62}', encoding="utf-8")

    assert module.publish_blocked_receipt(audit, viability) is True
    receipt = json.loads((module.BLOCKED_OUTPUT_DIR / "freeze_blocked_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["confirmation_performance_materialized"] is False
    assert receipt["frozen_spec_created"] is False
