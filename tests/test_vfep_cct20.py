import json

import pytest
import torch


def test_cct_mapping_places_empty_first_and_is_stable():
    from src.models.vfep_cct20 import canonical_cct_mapping

    categories = [{"id": 4, "name": "Coyote"}, {"id": 9, "name": "empty"}, {"id": 2, "name": "Bobcat"}]
    mapping, names, checksum = canonical_cct_mapping(categories)

    assert names == ("empty", "bobcat", "coyote")
    assert mapping == {9: 0, 2: 1, 4: 2}
    assert len(checksum) == 64


def test_cct_tpa_formula_uses_frozen_text_scale_and_prototype_scale():
    from src.models.vfep_cct20 import build_cct_tpa_logits

    features = torch.eye(3)
    text = torch.eye(3)
    prototypes = torch.eye(3)
    logits, diagnostics = build_cct_tpa_logits(features, text, prototypes, torch.ones(3, dtype=torch.bool), torch.tensor(0.0))

    assert torch.allclose(logits, 51.0 * torch.eye(3))
    assert diagnostics["prototype_scale"] == 50.0


def test_cct_split_and_trans_test_firewall(tmp_path):
    from src.models.vfep_cct20 import validate_cct_split_locations, verify_cct_stage, write_cct_receipt

    payload = validate_cct_split_locations({
        "trans_validation": ["v"],
        "trans_test": [f"t{index}" for index in range(9)],
    })
    assert payload["limited_cluster_uncertainty"] is True

    preregistration = {
        "vfep_eta_grid": [0, 0.25, 0.5, 1],
        "stp_eta_grid": [0, 0.25, 0.5, 1],
        "control_eta_policy": "reuse_selected_vfep_eta",
        "primary_metric_labels": [0, 1, 2],
        "tie_tolerance": 0.0001,
        "tie_break_rule": "smaller_eta",
        "dataset_snapshot_sha256": "dataset",
        "source_bundle_sha256": "source",
        "trans_validation_preflight_passed": False,
    }
    preregistration_path = tmp_path / "prereg.json"
    ledger_path = tmp_path / "ledger.json"
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    ledger_path.write_text(json.dumps({"completed_stages": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="preflight"):
        verify_cct_stage("trans_test", preregistration_path, ledger_path)

    preregistration["trans_validation_preflight_passed"] = True
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    ledger_path.write_text(json.dumps({"completed_stages": ["trans_test"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="already produced"):
        verify_cct_stage("trans_test", preregistration_path, ledger_path)


def test_cct_receipt_atomically_consumes_trans_test_stage(tmp_path):
    from src.models.vfep_cct20 import verify_cct_stage, write_cct_receipt

    preregistration = {
        "vfep_eta_grid": [0, 0.25, 0.5, 1],
        "stp_eta_grid": [0, 0.25, 0.5, 1],
        "control_eta_policy": "reuse_selected_vfep_eta",
        "primary_metric_labels": [0, 1, 2],
        "tie_tolerance": 0.0001,
        "tie_break_rule": "smaller_eta",
        "dataset_snapshot_sha256": "dataset",
        "source_bundle_sha256": "source",
        "trans_validation_preflight_passed": True,
    }
    preregistration_path = tmp_path / "prereg.json"
    ledger_path = tmp_path / "ledger.json"
    report_path = tmp_path / "report.json"
    receipt_path = tmp_path / "receipt.json"
    preregistration_path.write_text(json.dumps(preregistration), encoding="utf-8")
    ledger_path.write_text(json.dumps({"completed_stages": []}), encoding="utf-8")
    report_path.write_text(json.dumps({"macro_f1": 0.5}), encoding="utf-8")

    write_cct_receipt("trans_test", receipt_path, preregistration_path, report_path, ledger_path)

    assert json.loads(ledger_path.read_text(encoding="utf-8"))["completed_stages"] == ["trans_test"]
    with pytest.raises(ValueError, match="already produced"):
        verify_cct_stage("trans_test", preregistration_path, ledger_path)
