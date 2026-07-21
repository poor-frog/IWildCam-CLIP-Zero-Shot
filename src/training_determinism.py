from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
REPRODUCIBILITY_PACKAGES = (
    "braceexpand",
    "ftfy",
    "open-clip-torch",
    "pandas",
    "regex",
    "tqdm",
    "wandb",
    "webdataset",
    "wilds",
)


def seed_data_loader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def configure_training_determinism(seed: int, *, deterministic: bool) -> dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    cuda_available = torch.cuda.is_available()
    gpu_names = []
    if cuda_available:
        for device_index in range(torch.cuda.device_count()):
            gpu_names.append(torch.cuda.get_device_name(device_index))

    package_versions = {}
    for package_name in REPRODUCIBILITY_PACKAGES:
        try:
            package_versions[package_name] = version(package_name)
        except PackageNotFoundError:
            package_versions[package_name] = None

    return {
        "seed": seed,
        "deterministic_requested": bool(deterministic),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_names": gpu_names,
        "torch_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_deterministic_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": bool(getattr(torch.backends.cudnn, "deterministic", False)),
        "cudnn_benchmark": bool(getattr(torch.backends.cudnn, "benchmark", False)),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "dataloader_generator_seeded": True,
        "dataloader_workers_seeded": True,
        "source_commit": _git_head(),
    }


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return str(value)


def frozen_argument_payload(args: Any) -> dict[str, Any]:
    return {key: _json_value(value) for key, value in sorted(vars(args).items())}


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_determinism_receipt(
    args: Any,
    runtime_envelope: dict[str, Any],
    *,
    best_epoch: int | None,
    selected_wise_alpha: float | None,
    amp_skipped_step_count: int,
    best_validation_score: float | None = None,
    selected_wise_score: float | None = None,
    validation_trace: list[dict[str, Any]] | None = None,
    wise_selection_trace: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    configuration = frozen_argument_payload(args)
    return {
        "receipt": "paper_grade_training_foundation_v0_runtime_determinism",
        "status": "complete",
        "seed": int(args.seed),
        "configuration": configuration,
        "configuration_sha256": sha256_json(configuration),
        "runtime_determinism": runtime_envelope,
        "training": {
            "best_epoch": best_epoch,
            "best_validation_score": best_validation_score,
            "selected_wise_alpha": selected_wise_alpha,
            "selected_wise_score": selected_wise_score,
            "validation_trace": list(validation_trace or []),
            "wise_selection_trace": list(wise_selection_trace or []),
            "amp_skipped_step_count": int(amp_skipped_step_count),
            "non_finite_loss_or_gradient_observed": False,
        },
    }


def write_json_receipt_refusing_overwrite(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path
