from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.models.lesvi_confirmation import (
    deterministic_output_checksum,
    publish_lesvi_confirmation,
    verify_lesvi_confirmation_ready,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the one-shot LESVI-v0 internal confirmation after immutable guard verification.")
    parser.add_argument("--frozen-spec", type=Path, required=True)
    parser.add_argument("--confirmation-ledger", type=Path, required=True)
    parser.add_argument("--next-ledger", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--viability-report", type=Path, required=True)
    parser.add_argument("--class-mapping", type=Path, required=True)
    parser.add_argument("--prior-artifact", type=Path, required=True)
    parser.add_argument("--synthetic-verification", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-location", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu", "xla"), default="auto")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--wandb-project", default="PoorFrogs")
    parser.add_argument("--wandb-run-name", default="lesvi-v0-val-confirm-one-shot")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _hash_keys(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _runtime_args(args: argparse.Namespace, device: str) -> SimpleNamespace:
    return SimpleNamespace(
        model="ViT-B-16",
        train_dataset="IWildCam",
        val_dataset="IWildCamVal",
        template="iwildcam_template",
        data_location=str(args.data_location),
        load=str(args.checkpoint),
        batch_size=args.batch_size,
        workers=args.workers,
        device=device,
        seed=0,
        cache_dir=None,
        max_train_batches=None,
        max_eval_batches=None,
        no_data_parallel=False,
        num_ood_hp_examples=-1,
        class_balanced_ood=False,
    )


def _context_eligible(metadata, sequence_field_index: int):
    import torch
    from src.models.loo_bcpd import sequence_groups
    from src.models.stmp_adapter import metadata_group_key

    result = torch.zeros(len(metadata), dtype=torch.bool)
    for group in sequence_groups(metadata, sequence_field_index, len(metadata)):
        if len(group) >= 2 and metadata_group_key(metadata[group[0]], sequence_field_index) is not None:
            result[torch.tensor(group, dtype=torch.long)] = True
    return result


def _write_summary(path: Path, report: dict[str, object]) -> None:
    methods = report["methods"]
    promotion = report["promotion"]
    lines = [
        "# LESVI-v0 One-Shot Val-Confirm",
        "",
        f"- Promotion passed: `{promotion['passed']}`",
        f"- TPA macro-F1: `{methods['tpa']['overall']['macro_f1']:.6f}`",
        f"- STP-Mean macro-F1: `{methods['stp_mean']['overall']['macro_f1']:.6f}`",
        f"- LESVI macro-F1: `{methods['lesvi']['overall']['macro_f1']:.6f}`",
        f"- Rotation common coverage: `{report['rotation']['coverage']:.6f}`",
        "",
        "This is an internal preregistered confirmation, not a blind holdout.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _open_wandb_registry(args: argparse.Namespace, spec_sha256: str):
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("LESVI one-shot confirmation requires W&B as a durable replay registry.")
    import wandb

    run_id = f"lesvi-{spec_sha256[:24]}"
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        id=run_id,
        resume="allow",
        config={"method": "LESVI-v0", "one_shot": True, "lesvi_spec_sha256": spec_sha256},
    )
    if bool(run.summary.get("confirm/receipt_published", False)):
        run.finish()
        raise RuntimeError("W&B durable registry already contains a successful receipt for this LESVI specification.")
    return run


def _publish_wandb_receipt(run, report: dict[str, object], receipt_path: Path) -> None:
    import wandb

    run.log({
        "confirm/tpa_macro_f1": report["methods"]["tpa"]["overall"]["macro_f1"],
        "confirm/stp_mean_macro_f1": report["methods"]["stp_mean"]["overall"]["macro_f1"],
        "confirm/lesvi_macro_f1": report["methods"]["lesvi"]["overall"]["macro_f1"],
        "confirm/promotion_passed": report["promotion"]["passed"],
        "confirm/rotation_coverage": report["rotation"]["coverage"],
    })
    artifact = wandb.Artifact("lesvi-v0-confirmation-receipt", type="confirmation-receipt")
    artifact.add_file(str(receipt_path))
    run.log_artifact(artifact)
    run.summary["confirm/receipt_published"] = True
    run.finish()


def run_confirmation(args: argparse.Namespace, ready, wandb_run) -> tuple[Path, dict[str, object]]:
    import torch
    from wilds.common.data_loaders import get_eval_loader
    from wilds.datasets.wilds_dataset import WILDSSubset

    from src.device import resolve_device_choice
    from src.eval_tail_cache import (
        _metadata_rows,
        build_prototypes,
        build_train_dataset,
        default_logits,
        extract_features,
        prototype_logits,
        validate_stp_audit_class_mapping,
    )
    from src.models.clip_encoder import CLIPEncoder
    from src.models.coop import maybe_data_parallel, unwrap_model
    from src.models.flyp import get_cached_flyp_zeroshot_classifier
    from src.models.lesvi import (
        build_donor_event_rotation,
        build_lesvi_logits,
        build_location_context_supports,
        estimate_lesvi_prior,
        load_prior_artifact,
        with_visibility_variant,
    )
    from src.models.lesvi_evaluation import build_confirmation_report, build_metric_support, write_metric_support
    from src.models.stmp_adapter import (
        apply_sequence_consensus,
        metadata_fields_from_dataset,
        metadata_group_key,
        resolve_metadata_field_index,
    )
    from src.models.stp_audit_split import apply_normalized_loo_mean, build_location_audit_split
    from src.models.vfep import build_vfep_logits
    from src.train_coop import build_eval_dataset

    device = resolve_device_choice(args.device)
    runtime = _runtime_args(args, device)
    torch.manual_seed(0)
    model = maybe_data_parallel(CLIPEncoder.load(args.checkpoint).to(device), runtime)
    encoder = unwrap_model(model)
    train_data = build_train_dataset(runtime, encoder)
    val_data = build_eval_dataset("IWildCamVal", encoder, runtime, allow_ood_hp_subsample=False)
    fields = metadata_fields_from_dataset(val_data)
    sequence_index = resolve_metadata_field_index(fields, "auto", ["seq_id", "sequence_id", "sequence"])
    location_index = resolve_metadata_field_index(fields, "auto", ["location", "camera", "camera_id", "location_id"])
    if sequence_index is None or location_index is None:
        raise RuntimeError("LESVI confirmation requires sequence and location metadata.")
    spec = _load_json(args.frozen_spec)
    audit = _load_json(args.audit_manifest)
    if int(spec["sequence_field_index"]) != sequence_index or int(spec["location_field_index"]) != location_index:
        raise RuntimeError("Current metadata fields do not match the frozen LESVI specification.")
    if int(audit["sequence_field_index"]) != sequence_index or int(audit["location_field_index"]) != location_index:
        raise RuntimeError("Current metadata fields do not match the frozen Audit manifest.")

    train_labels_raw = torch.as_tensor(train_data.train_dataset.y_array).long().view(-1)
    train_metadata_raw = _metadata_rows(train_data.train_dataset.metadata_array)
    train_counts = torch.bincount(train_labels_raw, minlength=len(train_data.classnames))
    all_metadata = _metadata_rows(val_data.test_dataset.metadata_array)
    all_labels = torch.as_tensor(val_data.test_dataset.y_array).long().view(-1)
    split = build_location_audit_split(
        all_metadata,
        all_labels,
        train_counts,
        sequence_field_index=sequence_index,
        location_field_index=location_index,
    )
    observed_audit_digests = sorted(
        hashlib.sha256(f"20260716|{location}".encode("utf-8")).hexdigest()
        for location in split.audit_locations
    )
    if observed_audit_digests != sorted(str(value) for value in audit.get("audit_location_digests", [])):
        raise RuntimeError("Current location split does not reproduce the frozen Audit manifest.")
    confirm_indices = torch.where(split.confirm_mask)[0]
    if confirm_indices.numel() == 0:
        raise RuntimeError("Frozen split contains no Val-Confirm frames.")
    confirm_metadata = [all_metadata[index] for index in confirm_indices.tolist()]
    confirm_labels = all_labels[confirm_indices]
    confirm_locations = [split.location_keys[index] for index in confirm_indices.tolist()]
    if any(value is None for value in confirm_locations):
        raise RuntimeError("Val-Confirm contains a missing location after split filtering.")

    mapping = _load_json(args.class_mapping)
    ordered_mapping_sha = str(spec["ordered_class_mapping_sha256"])
    prior = load_prior_artifact(args.prior_artifact, expected_class_mapping_sha256=ordered_mapping_sha)
    recomputed_prior = estimate_lesvi_prior(
        train_labels_raw,
        train_metadata_raw,
        sequence_field_index=sequence_index,
        location_field_index=location_index,
        class_count=len(train_data.classnames),
        class_mapping_sha256=ordered_mapping_sha,
    )
    if not (
        torch.equal(prior.pi, recomputed_prior.pi)
        and torch.equal(prior.theta, recomputed_prior.theta)
        and torch.equal(prior.mu, recomputed_prior.mu)
    ):
        raise RuntimeError("Train-only LESVI prior does not match the frozen artifact.")
    if mapping.get("classnames") != list(train_data.classnames):
        raise RuntimeError("Current dataset class order does not match the frozen mapping.")

    rotation_seeds = tuple(int(value) for value in spec["rotation_seeds"])
    rotations = tuple(
        build_donor_event_rotation(
            confirm_metadata,
            sequence_field_index=sequence_index,
            location_field_index=location_index,
            seed=seed,
        )
        for seed in rotation_seeds
    )
    context_eligible = _context_eligible(confirm_metadata, sequence_index)
    common_rotation = context_eligible.clone()
    for rotation in rotations:
        common_rotation &= rotation.available_mask

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix="lesvi-confirm-", dir=args.output_dir.parent.resolve()))
    try:
        support = build_metric_support(
            confirm_labels,
            confirm_metadata,
            sequence_field_index=sequence_index,
            rotation_common_mask=common_rotation,
        )
        support_checksum = write_metric_support(temporary_dir / "confirm_metric_support.json", support, confirm_labels)

        confirm_subset = WILDSSubset(
            val_data.test_dataset.dataset,
            val_data.test_dataset.indices[confirm_indices.numpy()],
            val_data.test_dataset.transform,
        )
        confirm_loader = get_eval_loader("standard", confirm_subset, num_workers=args.workers, batch_size=args.batch_size)
        train_features = extract_features(model, train_data.train_loader, runtime, "LESVI train prototypes", is_train=True)
        confirm_features = extract_features(model, confirm_loader, runtime, "LESVI Val-Confirm features")
        if not torch.equal(confirm_features["labels"], confirm_labels):
            raise RuntimeError("Val-Confirm feature extraction changed label ordering.")
        extracted_sequences = [metadata_group_key(row, sequence_index) for row in confirm_features["metadata"]]
        expected_sequences = [metadata_group_key(row, sequence_index) for row in confirm_metadata]
        extracted_locations = [metadata_group_key(row, location_index) for row in confirm_features["metadata"]]
        expected_locations = [metadata_group_key(row, location_index) for row in confirm_metadata]
        if extracted_sequences != expected_sequences or extracted_locations != expected_locations:
            raise RuntimeError("Val-Confirm feature extraction changed metadata ordering.")
        prototypes, present = build_prototypes(train_features["features"], train_features["labels"], len(train_data.classnames))
        classifier = get_cached_flyp_zeroshot_classifier(runtime, encoder)
        current_mapping_sha = validate_stp_audit_class_mapping(train_data.classnames, classifier, prototypes)
        if current_mapping_sha != ordered_mapping_sha:
            raise RuntimeError("Classifier/prototype class order does not match the frozen mapping.")

        base = default_logits(confirm_features["features"], classifier)
        tpa = base + 50.0 * prototype_logits(confirm_features["features"], prototypes, present, beta=1.0)
        if not torch.isfinite(tpa).all():
            raise RuntimeError("Non-finite TPA logits invalidate LESVI confirmation.")
        stp_mean = apply_sequence_consensus(tpa, confirm_metadata, sequence_index, 0.5)
        lesvi = build_lesvi_logits(tpa, confirm_metadata, sequence_field_index=sequence_index, prior=prior)
        no_visibility = build_lesvi_logits(
            tpa,
            confirm_metadata,
            sequence_field_index=sequence_index,
            prior=with_visibility_variant(prior, "none"),
        )
        global_visibility = build_lesvi_logits(
            tpa,
            confirm_metadata,
            sequence_field_index=sequence_index,
            prior=with_visibility_variant(prior, "global"),
        )
        self_inclusive = build_lesvi_logits(tpa, confirm_metadata, sequence_field_index=sequence_index, prior=prior, include_target=True)
        location_supports = build_location_context_supports(
            confirm_metadata,
            sequence_field_index=sequence_index,
            location_field_index=location_index,
        )
        location_context = build_lesvi_logits(
            tpa,
            confirm_metadata,
            sequence_field_index=sequence_index,
            prior=prior,
            support_by_target=location_supports,
        )
        stp_loo = apply_normalized_loo_mean(tpa, confirm_metadata, sequence_field_index=sequence_index, eta=0.5)
        vfep = build_vfep_logits(tpa, confirm_metadata, sequence_field_index=sequence_index, strength=1.0).logits
        rotated_results = tuple(
            build_lesvi_logits(
                tpa,
                confirm_metadata,
                sequence_field_index=sequence_index,
                prior=prior,
                support_by_target=rotation.support_by_target,
            )
            for rotation in rotations
        )
        logits_by_name = {
            "tpa": tpa,
            "stp_mean": stp_mean,
            "lesvi": lesvi.logits,
            "no_visibility": no_visibility.logits,
            "stp_loo_diagnostic": stp_loo,
            "vfep_v0_diagnostic": vfep,
            "global_visibility_diagnostic": global_visibility.logits,
            "self_inclusive_diagnostic": self_inclusive.logits,
            "location_context_diagnostic": location_context.logits,
        }
        report = build_confirmation_report(
            labels=confirm_labels,
            logits_by_name=logits_by_name,
            support=support,
            location_keys=[str(value) for value in confirm_locations],
            rotation_logits=[result.logits for result in rotated_results],
            rotation_common_mask=common_rotation,
            context_eligible_mask=context_eligible,
            rotation_fallback_counts=[rotation.fallback_assignment_count for rotation in rotations],
        )
        fallback_summary = {
            "metadata_fallback_count": int(lesvi.metadata_fallback_mask.sum().item()),
            "numerical_fallback_count": int(lesvi.numerical_fallback_mask.sum().item()),
            "nonfinite_baseline_count": 0,
            "metadata_fallback_sample_keys_sha256": _hash_keys([str(index) for index in torch.where(lesvi.metadata_fallback_mask)[0].tolist()]),
            "numerical_fallback_sample_keys_sha256": _hash_keys([str(index) for index in torch.where(lesvi.numerical_fallback_mask)[0].tolist()]),
            "metadata_fallback_event_keys_sha256": _hash_keys([
                str(metadata_group_key(confirm_metadata[index], sequence_index))
                for index in torch.where(lesvi.metadata_fallback_mask)[0].tolist()
            ]),
            "numerical_fallback_event_keys_sha256": _hash_keys([
                str(metadata_group_key(confirm_metadata[index], sequence_index))
                for index in torch.where(lesvi.numerical_fallback_mask)[0].tolist()
            ]),
        }
        (temporary_dir / "lesvi_confirmation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (temporary_dir / "fallback_summary.json").write_text(json.dumps(fallback_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_summary(temporary_dir / "lesvi_confirmation_report.md", report)
        output_files = [path for path in temporary_dir.iterdir() if path.is_file()]
        output_checksum = deterministic_output_checksum(output_files)
        receipt_path = publish_lesvi_confirmation(
            temporary_output=temporary_dir,
            final_output=args.output_dir,
            ready=ready,
            previous_ledger_path=args.confirmation_ledger,
            next_ledger_path=args.next_ledger,
            git_commit=_git_commit(args.workspace_root),
            output_sha256=output_checksum,
            metric_support_sha256=support_checksum,
            fallback_summary=fallback_summary,
            wandb_run_id=str(wandb_run.id),
        )
    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        raise
    return receipt_path, report


def main(args: argparse.Namespace) -> None:
    ready = verify_lesvi_confirmation_ready(
        spec_path=args.frozen_spec,
        ledger_path=args.confirmation_ledger,
        workspace_root=args.workspace_root,
        audit_manifest_path=args.audit_manifest,
        viability_report_path=args.viability_report,
        class_mapping_path=args.class_mapping,
        prior_artifact_path=args.prior_artifact,
        synthetic_verification_path=args.synthetic_verification,
        checkpoint_path=args.checkpoint,
        final_output_path=args.output_dir,
    )
    print(f"LESVI confirmation guard passed for frozen specification {ready.spec_sha256}.")
    if args.validate_only:
        return
    wandb_run = _open_wandb_registry(args, ready.spec_sha256)
    try:
        receipt, report = run_confirmation(args, ready, wandb_run)
        _publish_wandb_receipt(wandb_run, report, receipt)
    except Exception:
        if not bool(wandb_run.summary.get("confirm/receipt_published", False)):
            wandb_run.finish(exit_code=1)
        raise
    print(f"LESVI confirmation published atomically with receipt {receipt}.")


if __name__ == "__main__":
    main(parse_arguments())
