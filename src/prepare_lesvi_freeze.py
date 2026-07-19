from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.models.stp_confirmation import build_source_bundle_checksum, file_sha256


SOURCE_FILES = (
    "clip/__init__.py",
    "clip/bpe_simple_vocab_16e6.txt.gz",
    "clip/clip.py",
    "clip/model.py",
    "clip/simple_tokenizer.py",
    "experiments/lesvi_v0/preregistration.json",
    "src/__init__.py",
    "src/config.py",
    "src/datasets/__init__.py",
    "src/datasets/dataloader.py",
    "src/datasets/iwildcam.py",
    "src/datasets/iwildcam_metadata/labels.csv",
    "src/device.py",
    "src/eval_lesvi_internal_confirm.py",
    "src/eval_drm_blend.py",
    "src/eval_tail_cache.py",
    "src/models/__init__.py",
    "src/models/clip_encoder.py",
    "src/models/btel.py",
    "src/models/btel_artifacts.py",
    "src/models/coop.py",
    "src/models/flyp.py",
    "src/models/lesvi.py",
    "src/models/lesvi_confirmation.py",
    "src/models/lesvi_evaluation.py",
    "src/models/logit_adjustment.py",
    "src/models/loo_bcpd.py",
    "src/models/eval.py",
    "src/models/maple_full.py",
    "src/models/maple_lora.py",
    "src/models/sctr.py",
    "src/models/stmp_adapter.py",
    "src/models/stp_audit_split.py",
    "src/models/stp_audit_metrics.py",
    "src/models/stp_audit_report.py",
    "src/models/stp_confirmation.py",
    "src/models/stp_diagnostics.py",
    "src/models/tail_prototype.py",
    "src/models/tip_adapter.py",
    "src/models/vfep.py",
    "src/models/vfep_pilot.py",
    "src/models/zeroshot.py",
    "src/prepare_lesvi_freeze.py",
    "src/report_lesvi_calibration.py",
    "src/templates/__init__.py",
    "src/templates/iwildcam.py",
    "src/templates/iwildcam_drm.py",
    "src/train_coop.py",
    "src/train_flyp.py",
    "src/train_maple_full.py",
)
EXPECTED_CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return payload


def _validate_inputs(
    audit: dict[str, object],
    mapping: dict[str, object],
    ledger: dict[str, object],
    viability: dict[str, object],
) -> tuple[list[str], str]:
    if audit.get("phase") != "val_audit_only" or audit.get("confirmation_performance_materialized") is not False:
        raise ValueError("LESVI freeze requires an unopened Val-Confirm audit manifest.")
    if audit.get("foundation") != "flyp" or str(audit.get("split_seed")) != "20260716":
        raise ValueError("LESVI-v0 requires the clean FLYP audit foundation and frozen 20260716 split.")
    viability_thresholds = (
        int(viability.get("audit_location_count", 0)) >= 10,
        int(viability.get("confirm_location_count", 0)) >= 10,
        int(viability.get("audit_supported_tail_class_count", 0)) >= 5,
        int(viability.get("confirm_supported_tail_class_count", 0)) >= 5,
        float(viability.get("confirm_supported_class_fraction", 0.0)) >= 0.8,
        float(viability.get("confirm_largest_location_frame_fraction", 1.0)) <= 0.5,
    )
    if viability.get("viability_pass") is not True or not all(viability_thresholds):
        raise ValueError("LESVI internal confirmation is unavailable because the frozen split failed viability preflight.")
    classnames = mapping.get("classnames")
    mapping_sha = mapping.get("class_mapping_sha256")
    if not isinstance(classnames, list) or not classnames or not all(isinstance(name, str) for name in classnames):
        raise ValueError("Class mapping must include ordered classnames.")
    if str(classnames[0]).strip().lower() != "empty" or mapping.get("empty_class_index") != 0:
        raise ValueError("LESVI requires canonical class 0 to be exactly 'empty'.")
    if not isinstance(mapping_sha, str) or not mapping_sha:
        raise ValueError("Class mapping must include class_mapping_sha256.")
    encoded = "\n".join(f"{index}:{name}" for index, name in enumerate(classnames)).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != mapping_sha:
        raise ValueError("Ordered class mapping checksum does not match classnames.")
    if ledger.get("confirmations") != []:
        raise ValueError("LESVI freeze requires an empty genesis confirmation ledger.")
    return classnames, mapping_sha


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate train-only LESVI priors and an immutable confirmation specification.")
    parser.add_argument("--data-location", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--viability-report", type=Path, required=True)
    parser.add_argument("--confirmation-genesis-ledger", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    import torch
    from src.models.lesvi import estimate_lesvi_prior, verify_lesvi_synthetic_reference, write_prior_artifact
    from src.models.stmp_adapter import metadata_fields_from_dataset, resolve_metadata_field_index

    audit = _load_json(args.audit_manifest, "audit manifest")
    mapping = _load_json(args.class_mapping, "class mapping")
    viability = _load_json(args.viability_report, "confirmation viability report")
    ledger = _load_json(args.confirmation_genesis_ledger, "confirmation genesis ledger")
    classnames, mapping_sha = _validate_inputs(audit, mapping, ledger, viability)
    if args.checkpoint.name != EXPECTED_CHECKPOINT_NAME:
        raise ValueError(f"LESVI-v0 requires the clean official checkpoint named {EXPECTED_CHECKPOINT_NAME}.")

    import wilds

    dataset = wilds.get_dataset(dataset="iwildcam", root_dir=str(args.data_location))
    train = dataset.get_subset("train", transform=None)
    fields = metadata_fields_from_dataset(train)
    sequence_index = resolve_metadata_field_index(fields, "auto", ["seq_id", "sequence_id", "sequence"])
    location_index = resolve_metadata_field_index(fields, "auto", ["location", "camera", "camera_id", "location_id"])
    if sequence_index is None or location_index is None:
        raise ValueError("Training metadata must expose sequence and location fields.")
    labels = torch.as_tensor(train.y_array).long().view(-1)
    dataset_class_count = int(getattr(dataset, "n_classes", len(classnames)))
    if dataset_class_count != len(classnames):
        raise ValueError("WILDS dataset class count does not match the frozen class mapping.")
    metadata = [torch.as_tensor(row) for row in train.metadata_array]
    prior = estimate_lesvi_prior(
        labels,
        metadata,
        sequence_field_index=sequence_index,
        location_field_index=location_index,
        class_count=len(classnames),
        class_mapping_sha256=mapping_sha,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prior_path = output_dir / "lesvi_prior.json"
    write_prior_artifact(prior_path, prior)
    synthetic = verify_lesvi_synthetic_reference()
    synthetic_path = output_dir / "synthetic_verification.json"
    synthetic_path.write_text(json.dumps(synthetic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_checksum = build_source_bundle_checksum(args.workspace_root.resolve(), SOURCE_FILES)
    spec = {
        "method": "LESVI-v0",
        "confirmation_status": "frozen",
        "source_files": list(SOURCE_FILES),
        "source_bundle_sha256": source_checksum,
        "audit_manifest_sha256": file_sha256(args.audit_manifest),
        "viability_report_sha256": file_sha256(args.viability_report),
        "class_mapping_sha256": file_sha256(args.class_mapping),
        "ordered_class_mapping_sha256": mapping_sha,
        "prior_artifact_sha256": file_sha256(prior_path),
        "synthetic_verification_sha256": file_sha256(synthetic_path),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "confirmation_ledger_genesis_sha256": file_sha256(args.confirmation_genesis_ledger),
        "sequence_field_index": sequence_index,
        "location_field_index": location_index,
        "preregistration_sha256": file_sha256(args.workspace_root / "experiments/lesvi_v0/preregistration.json"),
        "rotation_seeds": list(range(20260718, 20260817)),
        "bootstrap_samples": 2000,
        "bootstrap_seed": 20260718,
        "stp_eta": 0.5,
        "diagnostic_vfep_strength": 1.0,
        "validation_selected_strength": None,
        "val_audit_lesvi_executed": False,
        "id_ood_opened": False,
    }
    spec_path = output_dir / "lesvi_frozen_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"LESVI train-only prior written to {prior_path}")
    print(f"LESVI frozen specification written to {spec_path}")


if __name__ == "__main__":
    main(parse_arguments())
