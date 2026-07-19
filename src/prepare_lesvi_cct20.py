from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.models.lesvi import write_prior_artifact
from src.models.lesvi_cct20 import (
    cct_label_diagnostics,
    cct_snapshot_sha256,
    estimate_cct_lesvi_prior,
    file_sha256,
    load_cct_metadata_manifest,
    load_cct_records,
)
from src.models.stp_confirmation import build_source_bundle_checksum
from src.models.vfep_cct20 import validate_cct_split_locations


SOURCE_FILES = (
    "experiments/lesvi_v0/cct20_preregistration.json",
    "kaggle-cct20-lesvi-eval/kaggle_main.py",
    "src/eval_lesvi_cct20.py",
    "src/models/lesvi.py",
    "src/models/lesvi_cct20.py",
    "src/models/lesvi_evaluation.py",
    "src/models/loo_bcpd.py",
    "src/models/stmp_adapter.py",
    "src/models/stp_confirmation.py",
    "src/models/vfep_cct20.py",
    "src/prepare_lesvi_cct20.py",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze train-only LESVI-v0 priors and protocol for CCT-20.")
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--cis-validation-annotations", type=Path, required=True)
    parser.add_argument("--trans-validation-annotations", type=Path, required=True)
    parser.add_argument("--trans-test-annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def _prior_viability(prior) -> dict[str, object]:
    diagnostics = prior.diagnostics
    empty_event_count = int(diagnostics.get("valid_empty_event_count", 0))
    empty_frame_counts = [int(value) for value in diagnostics.get("empty_frame_counts", [])]
    species_with_observed_empty = sum(value > 0 for value in empty_frame_counts)
    checks = {
        "has_train_all_empty_event": empty_event_count > 0,
        "has_train_species_event_with_empty_frame": species_with_observed_empty > 0,
    }
    return {
        "viability_pass": all(checks.values()),
        "checks": checks,
        "valid_empty_event_count": empty_event_count,
        "species_with_observed_empty_frame_count": species_with_observed_empty,
        "reason_if_blocked": None if all(checks.values()) else "train_only_visibility_prior_not_identifiable",
        "trans_test_labels_opened": False,
    }


def main(args: argparse.Namespace) -> None:
    annotation_paths = {
        "train": args.train_annotations.resolve(),
        "cis_validation": args.cis_validation_annotations.resolve(),
        "trans_validation": args.trans_validation_annotations.resolve(),
        "trans_test": args.trans_test_annotations.resolve(),
    }
    loaded = {
        name: load_cct_records(path, args.image_root)
        for name, path in annotation_paths.items()
        if name != "trans_test"
    }
    train_records, class_names, mapping_sha = loaded["train"]
    for stage, (_, stage_names, stage_mapping_sha) in loaded.items():
        if stage_names != class_names or stage_mapping_sha != mapping_sha:
            raise ValueError(f"CCT class mapping differs between train and {stage}.")
    trans_test_manifest = load_cct_metadata_manifest(annotation_paths["trans_test"])
    if tuple(trans_test_manifest["class_names"]) != class_names or trans_test_manifest["class_mapping_sha256"] != mapping_sha:
        raise ValueError("CCT class mapping differs between train and trans_test.")
    split_locations = {
        "trans_validation": sorted({record.location for record in loaded["trans_validation"][0]}),
        "trans_test": list(trans_test_manifest["locations"]),
    }
    split_validation = validate_cct_split_locations(split_locations)
    prior = estimate_cct_lesvi_prior(train_records, len(class_names), mapping_sha)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = output_dir / "lesvi_cct20_prior.json"
    write_prior_artifact(prior_path, prior)
    mapping_path = output_dir / "lesvi_cct20_class_mapping.json"
    mapping_path.write_text(json.dumps({
        "class_names": list(class_names),
        "class_mapping_sha256": mapping_sha,
        "empty_class_index": 0,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = output_dir / "lesvi_cct20_split_manifest.json"
    manifest_path.write_text(json.dumps({
        "annotation_sha256": {name: file_sha256(path) for name, path in annotation_paths.items()},
        "dataset_snapshot_sha256": cct_snapshot_sha256(tuple(annotation_paths.values())),
        "record_counts": {
            **{name: len(records) for name, (records, _, _) in loaded.items()},
            "trans_test": int(trans_test_manifest["image_count"]),
        },
        "single_label_policy": "exclude_images_with_multiple_unique_category_ids",
        "label_diagnostics": {
            name: cct_label_diagnostics(path)
            for name, path in annotation_paths.items()
            if name != "trans_test"
        },
        "trans_test_label_diagnostics_opened": False,
        "split_locations": split_locations,
        **split_validation,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    viability = _prior_viability(prior)
    viability_path = output_dir / "lesvi_cct20_prior_viability.json"
    viability_path.write_text(json.dumps(viability, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if viability["viability_pass"] is not True:
        blocked_path = output_dir / "lesvi_cct20_freeze_blocked_receipt.json"
        blocked_path.write_text(json.dumps({
            "method": "LESVI-CCT20-v0",
            "status": "blocked",
            "reason": viability["reason_if_blocked"],
            "frozen_spec_created": False,
            "trans_validation_opened": False,
            "trans_test_opened": False,
            "prior_artifact_sha256": file_sha256(prior_path),
            "split_manifest_sha256": file_sha256(manifest_path),
            "viability_report_sha256": file_sha256(viability_path),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"CCT-20 LESVI freeze blocked; receipt written to {blocked_path}")
        return
    preregistration_path = args.workspace_root / "experiments/lesvi_v0/cct20_preregistration.json"
    spec = {
        "method": "LESVI-CCT20-v0",
        "confirmation_status": "frozen",
        "dataset_snapshot_sha256": cct_snapshot_sha256(tuple(annotation_paths.values())),
        "annotation_sha256": {name: file_sha256(path) for name, path in annotation_paths.items()},
        "class_mapping_sha256": mapping_sha,
        "class_mapping_artifact_sha256": file_sha256(mapping_path),
        "prior_artifact_sha256": file_sha256(prior_path),
        "split_manifest_sha256": file_sha256(manifest_path),
        "prior_viability_sha256": file_sha256(viability_path),
        "preregistration_sha256": file_sha256(preregistration_path),
        "source_files": list(SOURCE_FILES),
        "source_bundle_sha256": build_source_bundle_checksum(args.workspace_root.resolve(), SOURCE_FILES),
        "stp_eta": 0.5,
        "rotation_seeds": list(range(20260718, 20260817)),
        "trans_test_opened": False,
    }
    spec_path = output_dir / "lesvi_cct20_frozen_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_path = output_dir / "lesvi_cct20_ledger.json"
    ledger_path.write_text(json.dumps({
        "method": "LESVI-CCT20-v0",
        "frozen_spec_sha256": file_sha256(spec_path),
        "completed_stages": [],
        "receipts": [],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CCT-20 LESVI frozen specification written to {spec_path}")


if __name__ == "__main__":
    main(parse_arguments())
