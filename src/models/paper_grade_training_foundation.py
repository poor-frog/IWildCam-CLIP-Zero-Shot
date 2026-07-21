from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.training_determinism import frozen_argument_payload, sha256_json


PILOT_SEEDS = frozenset({20260721, 20260722, 20260723})
COMPLETION_SEEDS = frozenset({20260724, 20260725})
VALIDATION_SEEDS = PILOT_SEEDS | COMPLETION_SEEDS
VALIDATION_ALLOWED_DATASETS = frozenset({"IWildCam", "IWildCamVal"})
VALIDATION_FORBIDDEN_DATASETS = frozenset({"IWildCamIDVal", "IWildCamID", "IWildCamOOD", "CCT-20"})
FROZEN_WISE_ALPHAS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)
FROZEN_VALIDATION_ARGUMENTS = {
    "model": "ViT-B-16",
    "train_dataset": "IWildCam",
    "val_dataset": "IWildCamVal",
    "template": "iwildcam_template",
    "epochs": 20,
    "batch_size": 256,
    "workers": 4,
    "lr": 0.00001,
    "wd": 0.2,
    "lr_scheduler": "cosine",
    "warmup_length": 0,
    "maple_precision": "amp",
    "best_metric": "F1-macro_all",
    "drm_weight": 0.0,
    "drm_warmup_epochs": 0,
    "tail_proto_weight": 0.0,
    "btel_weight": 0.0,
    "wise_alphas": "0,0.05,0.1,0.15,0.2,0.3",
    "wise_eval_alpha": None,
    "load": None,
    "eval_datasets": None,
    "max_train_batches": None,
    "max_eval_batches": None,
    "num_ood_hp_examples": -1,
    "class_balanced_ood": False,
    "cd_path": None,
    "no_load_best_for_eval": False,
    "btel_audit_only": False,
}
RUN_SPECIFIC_CONFIGURATION_FIELDS = frozenset(
    {
        "best_checkpoint",
        "determinism_receipt",
        "save",
        "seed",
        "wandb_run_name",
    }
)


@dataclass(frozen=True)
class PaperGradeRunPaths:
    run_directory: Path
    best_checkpoint: Path
    final_checkpoint: Path
    receipt: Path


class ValidationSplitFirewall:
    def __init__(self) -> None:
        self._accessed: list[str] = []

    def record(self, dataset_name: str) -> None:
        if dataset_name not in VALIDATION_ALLOWED_DATASETS:
            raise ValueError(
                "Paper-Grade Training Foundation v0 validation firewall blocked "
                f"dataset {dataset_name!r}; only IWildCam train and IWildCamVal are allowed."
            )
        if dataset_name not in self._accessed:
            self._accessed.append(dataset_name)

    def receipt_payload(self) -> dict[str, Any]:
        return {
            "stage": "five_seed_validation",
            "allowed_datasets": sorted(VALIDATION_ALLOWED_DATASETS),
            "forbidden_datasets": sorted(VALIDATION_FORBIDDEN_DATASETS),
            "accessed_datasets": list(self._accessed),
            "forbidden_dataset_accessed": False,
            "passed": set(self._accessed) <= VALIDATION_ALLOWED_DATASETS,
        }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def build_source_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    candidates = [
        root / "pyproject.toml",
        root / "experiments" / "paper_grade_training_foundation_v0" / "preregistration.json",
        root / "experiments" / "paper_grade_training_foundation_v0" / "preregistration.schema.json",
    ]
    candidates.extend(sorted((root / "src").rglob("*.py")))
    files = [path for path in candidates if path.is_file() and "__pycache__" not in path.parts]
    manifest = [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)} for path in files]
    return {
        "git_commit": _git_head(root),
        "source_tree_sha256": sha256_json(manifest),
        "hashed_file_count": len(manifest),
        "hashed_files": manifest,
    }


def _array_provenance(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def build_iwildcam_dataset_provenance(dataset: Any, data_location: str | Path) -> dict[str, Any]:
    data_dir_value = getattr(dataset, "_data_dir", None)
    data_dir = Path(data_dir_value).resolve() if data_dir_value is not None else None
    metadata_path = data_dir / "metadata.csv" if data_dir is not None else None
    metadata = None
    if metadata_path is not None and metadata_path.is_file():
        metadata = {
            "path": str(metadata_path),
            "size_bytes": metadata_path.stat().st_size,
            "sha256": sha256_file(metadata_path),
        }

    split_dict = {str(key): int(value) for key, value in sorted(getattr(dataset, "split_dict", {}).items())}
    payload = {
        "dataset": "iwildcam",
        "dataset_class": f"{dataset.__class__.__module__}.{dataset.__class__.__qualname__}",
        "data_location": str(Path(data_location).resolve()),
        "data_directory": str(data_dir) if data_dir is not None else None,
        "dataset_version": str(getattr(dataset, "version", None)) if getattr(dataset, "version", None) is not None else None,
        "n_classes": int(getattr(dataset, "n_classes", 0)) or None,
        "split_dict": split_dict,
        "split_array": _array_provenance(getattr(dataset, "split_array", None)),
        "label_array": _array_provenance(getattr(dataset, "y_array", None)),
        "metadata_csv": metadata,
    }
    payload["dataset_snapshot_sha256"] = sha256_json(payload)
    return payload


def protocol_configuration(args: Any) -> dict[str, Any]:
    configuration = frozen_argument_payload(args)
    return {key: value for key, value in configuration.items() if key not in RUN_SPECIFIC_CONFIGURATION_FIELDS}


def _require_seed_path(path: Path, seed: int, label: str) -> None:
    if str(seed) not in str(path):
        raise ValueError(f"{label} must contain seed {seed}: {path}")


def validate_frozen_validation_arguments(args: Any) -> None:
    mismatches = []
    for field, expected in FROZEN_VALIDATION_ARGUMENTS.items():
        observed = getattr(args, field, "<missing>")
        if observed != expected:
            mismatches.append(f"{field}={observed!r} (expected {expected!r})")
    if mismatches:
        raise ValueError("Paper-grade frozen configuration mismatch: " + "; ".join(mismatches))


def validate_paper_grade_validation_run(args: Any, best_checkpoint_path: str | Path) -> PaperGradeRunPaths:
    seed = int(args.seed)
    if seed not in VALIDATION_SEEDS:
        raise ValueError(f"Seed {seed} is not in the preregistered five-seed validation set.")
    if not bool(getattr(args, "deterministic_training", False)):
        raise ValueError("Paper-grade validation runs require --deterministic-training.")
    if getattr(args, "load", None) is not None:
        raise ValueError("Paper-grade validation runs must start from fresh pretrained weights; --load is forbidden.")
    if getattr(args, "train_dataset", None) != "IWildCam":
        raise ValueError("Paper-grade validation runs require --train-dataset=IWildCam.")
    if getattr(args, "val_dataset", None) != "IWildCamVal":
        raise ValueError("Paper-grade validation runs require --val-dataset=IWildCamVal.")

    eval_datasets = set(getattr(args, "eval_datasets", None) or [])
    forbidden_eval = eval_datasets - {"IWildCamVal"}
    if forbidden_eval:
        raise ValueError(f"Paper-grade validation firewall blocked eval datasets: {sorted(forbidden_eval)}")
    validate_frozen_validation_arguments(args)

    receipt_value = getattr(args, "determinism_receipt", None)
    save_value = getattr(args, "save", None)
    if receipt_value is None or save_value is None:
        raise ValueError("Paper-grade validation runs require explicit --determinism-receipt and --save paths.")

    receipt = Path(receipt_value).resolve()
    final_checkpoint = Path(save_value).resolve()
    best_checkpoint = Path(best_checkpoint_path).resolve()
    if not final_checkpoint.suffix:
        raise ValueError("Paper-grade --save must be an explicit checkpoint file path, not a directory.")
    for path, label in (
        (receipt, "Receipt path"),
        (final_checkpoint, "Final checkpoint path"),
        (best_checkpoint, "Best checkpoint path"),
    ):
        _require_seed_path(path, seed, label)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite paper-grade artifact: {path}")

    parents = {receipt.parent, final_checkpoint.parent, best_checkpoint.parent}
    if len(parents) != 1:
        raise ValueError("Receipt, best checkpoint, and final checkpoint must share one immutable per-seed run directory.")
    if len({receipt, final_checkpoint, best_checkpoint}) != 3:
        raise ValueError("Receipt, best checkpoint, and final checkpoint paths must be distinct.")
    run_directory = receipt.parent
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(f"Immutable per-seed run directory is not empty: {run_directory}")
    return PaperGradeRunPaths(run_directory, best_checkpoint, final_checkpoint, receipt)


def build_checkpoint_provenance(paths: PaperGradeRunPaths) -> dict[str, Any]:
    artifacts = {}
    for label, path in (("best_validation", paths.best_checkpoint), ("final_wise", paths.final_checkpoint)):
        if not path.is_file():
            raise FileNotFoundError(f"Required paper-grade checkpoint was not created: {path}")
        artifacts[label] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def enrich_paper_grade_run_receipt(
    receipt: dict[str, Any],
    args: Any,
    *,
    source_provenance: dict[str, Any],
    dataset_provenance: dict[str, Any],
    checkpoint_provenance: dict[str, Any],
    split_firewall: dict[str, Any],
) -> dict[str, Any]:
    protocol_config = protocol_configuration(args)
    receipt = dict(receipt)
    receipt["receipt"] = "paper_grade_training_foundation_v0_validation_run"
    receipt["protocol_configuration"] = protocol_config
    receipt["protocol_configuration_sha256"] = sha256_json(protocol_config)
    receipt["provenance"] = {
        "source": source_provenance,
        "dataset": dataset_provenance,
        "checkpoints": checkpoint_provenance,
    }
    receipt["split_firewall"] = split_firewall
    training = receipt.get("training", {})
    validation_trace = training.get("validation_trace", [])
    wise_trace = training.get("wise_selection_trace", [])
    best_epoch = training.get("best_epoch")
    best_score = training.get("best_validation_score")
    selected_alpha = training.get("selected_wise_alpha")
    selected_score = training.get("selected_wise_score")
    validation_scores = [item.get("metric_value") for item in validation_trace]
    wise_scores = [item.get("metric_value") for item in wise_trace]
    finite_validation = len(validation_scores) == 20 and all(
        isinstance(score, (int, float)) and math.isfinite(score) for score in validation_scores
    )
    finite_wise = len(wise_scores) == len(FROZEN_WISE_ALPHAS) and all(
        isinstance(score, (int, float)) and math.isfinite(score) for score in wise_scores
    )
    selection_complete = (
        isinstance(best_epoch, int)
        and 1 <= best_epoch <= 20
        and [item.get("epoch") for item in validation_trace] == list(range(1, 21))
        and finite_validation
        and isinstance(best_score, (int, float))
        and math.isfinite(best_score)
        and math.isclose(best_score, max(validation_scores))
        and selected_alpha in FROZEN_WISE_ALPHAS
        and [item.get("alpha") for item in wise_trace] == list(FROZEN_WISE_ALPHAS)
        and finite_wise
        and isinstance(selected_score, (int, float))
        and math.isfinite(selected_score)
        and math.isclose(selected_score, max(wise_scores))
    )
    receipt["validity"] = {
        "checkpoint_hashes_present": all(item.get("sha256") for item in checkpoint_provenance.values()),
        "source_hash_present": bool(source_provenance.get("source_tree_sha256")),
        "dataset_hash_present": bool(dataset_provenance.get("dataset_snapshot_sha256")),
        "split_firewall_passed": split_firewall.get("passed") is True,
        "frozen_configuration_match": True,
        "selection_complete": selection_complete,
        "zero_amp_skipped_steps": training.get("amp_skipped_step_count") == 0,
        "no_non_finite_event": training.get("non_finite_loss_or_gradient_observed") is False,
    }
    if not all(receipt["validity"].values()):
        receipt["status"] = "invalid"
    return receipt
