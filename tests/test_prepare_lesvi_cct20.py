import json
from types import SimpleNamespace

from src.prepare_lesvi_cct20 import main


def _annotation(path, locations, *, include_annotations=True):
    images = []
    annotations = []
    for index, location in enumerate(locations):
        image_id = f"{location}-{index}"
        images.append({
            "id": image_id,
            "file_name": f"{image_id}.jpg",
            "location": location,
            "seq_id": f"sequence-{index}",
        })
        if include_annotations:
            annotations.append({"id": f"a-{index}", "image_id": image_id, "category_id": 2})
    payload = {
        "categories": [{"id": 1, "name": "empty"}, {"id": 2, "name": "bobcat"}],
        "images": images,
    }
    if include_annotations:
        payload["annotations"] = annotations
    path.write_text(json.dumps(payload), encoding="utf-8")


def _viable_train_annotation(path):
    payload = {
        "categories": [{"id": 1, "name": "empty"}, {"id": 2, "name": "bobcat"}],
        "images": [
            {"id": "e0", "file_name": "e0.jpg", "location": "train", "seq_id": "empty"},
            {"id": "e1", "file_name": "e1.jpg", "location": "train", "seq_id": "empty"},
            {"id": "m0", "file_name": "m0.jpg", "location": "train", "seq_id": "mixed"},
            {"id": "m1", "file_name": "m1.jpg", "location": "train", "seq_id": "mixed"},
        ],
        "annotations": [
            {"id": "a0", "image_id": "e0", "category_id": 1},
            {"id": "a1", "image_id": "e1", "category_id": 1},
            {"id": "a2", "image_id": "m0", "category_id": 2},
            {"id": "a3", "image_id": "m1", "category_id": 1},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_prepare_cct_freezes_without_consuming_trans_test_labels(tmp_path):
    train = tmp_path / "train.json"
    cis = tmp_path / "cis.json"
    trans_validation = tmp_path / "trans-validation.json"
    trans_test = tmp_path / "trans-test.json"
    _viable_train_annotation(train)
    _annotation(cis, ["cis"])
    _annotation(trans_validation, ["tv"])
    _annotation(trans_test, [f"tt-{index}" for index in range(9)], include_annotations=False)
    output = tmp_path / "output"

    main(SimpleNamespace(
        train_annotations=train,
        cis_validation_annotations=cis,
        trans_validation_annotations=trans_validation,
        trans_test_annotations=trans_test,
        image_root=tmp_path,
        output_dir=output,
        workspace_root=__import__("pathlib").Path.cwd(),
    ))

    manifest = json.loads((output / "lesvi_cct20_split_manifest.json").read_text(encoding="utf-8"))
    spec = json.loads((output / "lesvi_cct20_frozen_spec.json").read_text(encoding="utf-8"))
    ledger = json.loads((output / "lesvi_cct20_ledger.json").read_text(encoding="utf-8"))
    viability = json.loads((output / "lesvi_cct20_prior_viability.json").read_text(encoding="utf-8"))
    assert manifest["record_counts"]["trans_test"] == 9
    assert manifest["trans_test_location_count"] == 9
    assert ledger["frozen_spec_sha256"]
    assert ledger["completed_stages"] == []
    assert viability["viability_pass"] is True
    assert spec["trans_test_opened"] is False


def test_prepare_cct_blocks_when_train_only_visibility_is_not_identifiable(tmp_path):
    train = tmp_path / "train.json"
    cis = tmp_path / "cis.json"
    trans_validation = tmp_path / "trans-validation.json"
    trans_test = tmp_path / "trans-test.json"
    _annotation(train, ["train"])
    _annotation(cis, ["cis"])
    _annotation(trans_validation, ["tv"])
    _annotation(trans_test, [f"tt-{index}" for index in range(9)], include_annotations=False)
    output = tmp_path / "blocked"

    main(SimpleNamespace(
        train_annotations=train,
        cis_validation_annotations=cis,
        trans_validation_annotations=trans_validation,
        trans_test_annotations=trans_test,
        image_root=tmp_path,
        output_dir=output,
        workspace_root=__import__("pathlib").Path.cwd(),
    ))

    receipt = json.loads((output / "lesvi_cct20_freeze_blocked_receipt.json").read_text(encoding="utf-8"))
    assert receipt["status"] == "blocked"
    assert receipt["reason"] == "train_only_visibility_prior_not_identifiable"
    assert receipt["trans_test_opened"] is False
    assert not (output / "lesvi_cct20_frozen_spec.json").exists()
