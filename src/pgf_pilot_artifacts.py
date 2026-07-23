from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PILOT_SEEDS = (20260721, 20260722, 20260723)
FROZEN_WISE_ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
EXPECTED_PACKAGE_VERSIONS = {
    "braceexpand": "0.1.7",
    "ftfy": "6.3.1",
    "open-clip-torch": "3.3.0",
    "pandas": "2.2.3",
    "regex": "2024.11.6",
    "tqdm": "4.67.3",
    "wandb": "0.28.0",
    "webdataset": "1.0.2",
    "wilds": "2.0.0",
}
EXPECTED_AMP_POLICY = {
    "autocast_enabled": True,
    "grad_scaler_enabled": True,
    "grad_scaler_initial_scale": 1.0,
    "grad_scaler_growth_interval": 2**31 - 1,
    "paper_grade_static_unit_scale": True,
}
CHECKPOINT_FILENAMES = {
    "best_validation": "best_checkpoint.pt",
    "final_wise": "final_checkpoint.pt",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_complete_trace(training: dict[str, Any], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validation_trace = training.get("validation_trace", [])
    wise_trace = training.get("wise_selection_trace", [])
    if [item.get("epoch") for item in validation_trace] != list(range(1, 21)):
        raise RuntimeError(f"Seed {seed} validation trace does not cover all 20 epochs.")
    if [item.get("alpha") for item in wise_trace] != list(FROZEN_WISE_ALPHAS):
        raise RuntimeError(f"Seed {seed} WiSE trace does not match the frozen grid.")
    return validation_trace, wise_trace


def _verify_checkpoints(
    run_directory: Path,
    checkpoint_provenance: dict[str, Any],
    seed: int,
) -> dict[str, str]:
    hashes = {}
    for role, filename in CHECKPOINT_FILENAMES.items():
        artifact = checkpoint_provenance.get(role)
        if not isinstance(artifact, dict):
            raise RuntimeError(f"Seed {seed} receipt is missing {role} checkpoint provenance.")
        if Path(str(artifact.get("path", ""))).name != filename:
            raise RuntimeError(f"Seed {seed} receipt has the wrong filename for {role}.")
        checkpoint_path = run_directory / filename
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Seed {seed} output is missing {checkpoint_path}.")
        observed_hash = sha256_file(checkpoint_path)
        if observed_hash != artifact.get("sha256"):
            raise RuntimeError(f"Seed {seed} checkpoint hash mismatch: {checkpoint_path}")
        hashes[role] = observed_hash
    return hashes


def verify_seed_run(run_directory: str | Path, seed: int, source_commit: str) -> dict[str, Any]:
    run_directory = Path(run_directory).resolve()
    seed = int(seed)
    if seed not in PILOT_SEEDS:
        raise ValueError(f"Seed {seed} is not in the preregistered pilot set.")
    if run_directory.name != f"seed-{seed}":
        nested = run_directory / f"seed-{seed}"
        if nested.is_dir():
            run_directory = nested
        else:
            raise ValueError(f"Seed {seed} output directory must be named seed-{seed}: {run_directory}")

    receipt_path = run_directory / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("seed") != seed:
        raise RuntimeError(f"Seed {seed} receipt is incomplete or has the wrong seed.")

    provenance = receipt.get("provenance", {})
    source = provenance.get("source", {})
    if source.get("git_commit") != source_commit:
        raise RuntimeError(f"Seed {seed} receipt is not bound to frozen source {source_commit}.")
    firewall = receipt.get("split_firewall", {})
    if firewall.get("accessed_datasets") != ["IWildCam", "IWildCamVal"]:
        raise RuntimeError(f"Seed {seed} accessed data outside train plus IWildCamVal.")
    validity = receipt.get("validity", {})
    if not validity or not all(validity.values()):
        raise RuntimeError(f"Seed {seed} receipt failed its provenance or firewall validity checks.")

    training = receipt.get("training", {})
    if training.get("amp_skipped_step_count") != 0:
        raise RuntimeError(f"Seed {seed} had skipped AMP optimizer steps.")
    if training.get("non_finite_loss_or_gradient_observed") is not False:
        raise RuntimeError(f"Seed {seed} recorded a non-finite loss or gradient event.")
    if not isinstance(training.get("best_epoch"), int) or not 1 <= training["best_epoch"] <= 20:
        raise RuntimeError(f"Seed {seed} does not have exactly one valid best epoch.")
    if training.get("selected_wise_alpha") not in FROZEN_WISE_ALPHAS:
        raise RuntimeError(f"Seed {seed} does not have one alpha from the frozen WiSE grid.")
    validation_trace, wise_trace = _require_complete_trace(training, seed)

    runtime = receipt.get("runtime_determinism", {})
    if runtime.get("package_versions") != EXPECTED_PACKAGE_VERSIONS:
        raise RuntimeError(f"Seed {seed} package versions differ from the pinned environment.")
    if runtime.get("amp_policy") != EXPECTED_AMP_POLICY:
        raise RuntimeError(f"Seed {seed} did not use the frozen paper-grade AMP policy.")

    checkpoint_hashes = _verify_checkpoints(run_directory, provenance.get("checkpoints", {}), seed)
    runtime_environment = {key: value for key, value in runtime.items() if key not in {"seed", "source_commit"}}
    return {
        "seed": seed,
        "receipt_path": f"seed-{seed}/run_receipt.json",
        "receipt_sha256": sha256_file(receipt_path),
        "source_commit": source_commit,
        "source_tree_sha256": source["source_tree_sha256"],
        "dataset_snapshot_sha256": provenance["dataset"]["dataset_snapshot_sha256"],
        "protocol_configuration_sha256": receipt["protocol_configuration_sha256"],
        "runtime_environment_sha256": sha256_json(runtime_environment),
        "best_checkpoint_sha256": checkpoint_hashes["best_validation"],
        "final_checkpoint_sha256": checkpoint_hashes["final_wise"],
        "best_epoch": training["best_epoch"],
        "best_validation_score": training["best_validation_score"],
        "selected_wise_alpha": training["selected_wise_alpha"],
        "selected_wise_score": training["selected_wise_score"],
        "amp_skipped_step_count": training["amp_skipped_step_count"],
        "non_finite_loss_or_gradient_observed": training["non_finite_loss_or_gradient_observed"],
        "validation_trace_sha256": sha256_json(validation_trace),
        "wise_selection_trace_sha256": sha256_json(wise_trace),
    }


def build_single_seed_manifest(seed_receipt: dict[str, Any]) -> dict[str, Any]:
    seed = seed_receipt.get("seed")
    if seed not in PILOT_SEEDS:
        raise RuntimeError(f"Single-seed manifest has an invalid pilot seed: {seed}")
    return {
        "manifest": "paper_grade_training_foundation_v0_single_seed",
        "status": "complete",
        "source_commit": seed_receipt["source_commit"],
        "seed": seed,
        "selection_split": "IWildCamVal",
        "final_splits_opened": False,
        "seed_receipt": seed_receipt,
    }


def build_pilot_manifest(seed_receipts: list[dict[str, Any]], source_commit: str) -> dict[str, Any]:
    if [item.get("seed") for item in seed_receipts] != list(PILOT_SEEDS):
        raise RuntimeError("Pilot manifest must contain the three preregistered seeds in order.")
    if any(item.get("source_commit") != source_commit for item in seed_receipts):
        raise RuntimeError("Pilot seed receipts do not match the frozen source commit.")
    for field in (
        "source_tree_sha256",
        "dataset_snapshot_sha256",
        "protocol_configuration_sha256",
        "runtime_environment_sha256",
    ):
        if len({item[field] for item in seed_receipts}) != 1:
            raise RuntimeError(f"Pilot seeds disagree on {field}.")
    for field in ("best_checkpoint_sha256", "final_checkpoint_sha256"):
        if len({item[field] for item in seed_receipts}) != len(PILOT_SEEDS):
            raise RuntimeError(f"Pilot seed checkpoints are not unique for {field}.")
    for item in seed_receipts:
        if item.get("amp_skipped_step_count") != 0:
            raise RuntimeError(f"Pilot seed {item.get('seed')} had skipped AMP optimizer steps.")
        if item.get("non_finite_loss_or_gradient_observed") is not False:
            raise RuntimeError(f"Pilot seed {item.get('seed')} recorded a non-finite event.")
    return {
        "manifest": "paper_grade_training_foundation_v0_three_seed_pilot",
        "status": "complete",
        "source_commit": source_commit,
        "seeds": list(PILOT_SEEDS),
        "selection_split": "IWildCamVal",
        "final_splits_opened": False,
        "seed_receipts": seed_receipts,
    }


def write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def aggregate_pilot_outputs(
    seed_output_directories: dict[int, str | Path],
    source_commit: str,
    output_path: str | Path,
) -> Path:
    if set(seed_output_directories) != set(PILOT_SEEDS):
        raise ValueError(f"Aggregation requires exactly the pilot seeds {PILOT_SEEDS}.")
    receipts = [
        verify_seed_run(seed_output_directories[seed], seed, source_commit)
        for seed in PILOT_SEEDS
    ]
    return write_json_exclusive(output_path, build_pilot_manifest(receipts, source_commit))


def _parse_seed_output(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected SEED=OUTPUT_DIRECTORY.")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid seed: {seed_text}") from error
    return seed, Path(path_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and aggregate the three PGF v0 pilot seed outputs.")
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--seed-output", action="append", type=_parse_seed_output, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_outputs = dict(args.seed_output)
    output = aggregate_pilot_outputs(seed_outputs, args.source_commit, args.output)
    print(f"Wrote immutable PGF pilot manifest to {output}")


if __name__ == "__main__":
    main()
