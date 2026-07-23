from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.pgf_pilot_artifacts import (
    EXPECTED_PACKAGE_VERSIONS,
    PILOT_SEEDS,
    build_single_seed_manifest,
    verify_seed_run,
    write_json_exclusive,
)


PINNED_DEPENDENCIES = tuple(
    f"{package}=={version}" for package, version in EXPECTED_PACKAGE_VERSIONS.items()
)


def run(command: list[str | Path], *, cwd: str | Path | None = None, env: dict[str, str] | None = None) -> None:
    printable = [str(part) for part in command]
    print("+", " ".join(printable), flush=True)
    subprocess.check_call(printable, cwd=cwd, env=env)


def ensure_dependencies(repository: Path) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *PINNED_DEPENDENCIES])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", repository, "--no-deps"])


def find_iwildcam_source_root() -> Path:
    candidates = [
        Path("/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"),
        Path("/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        for candidate in (root, root / "iwildcam_v2.0", root / "archive" / "iwildcam_v2.0"):
            if (candidate / "metadata.csv").is_file():
                return candidate
        matches = sorted(root.rglob("metadata.csv"))
        if matches:
            return matches[0].parent
    raise FileNotFoundError("Could not locate the attached IWildCam v2.0 dataset.")


def prepare_iwildcam_layout(repository: Path) -> Path:
    source_root = find_iwildcam_source_root()
    target_root = repository / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in source_root.iterdir():
        target_path = target_root / source_path.name
        if not target_path.exists() and not target_path.is_symlink():
            target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    return target_root.parent


def build_command(seed: int, repository: Path, output_root: Path, data_location: Path) -> list[str | Path]:
    if seed not in PILOT_SEEDS:
        raise ValueError(f"Seed {seed} is not in the preregistered pilot set.")
    run_directory = output_root / f"seed-{seed}"
    return [
        sys.executable,
        "src/train_flyp.py",
        "--paper-grade-training-foundation-v0",
        "--deterministic-training",
        f"--determinism-receipt={run_directory / 'run_receipt.json'}",
        f"--save={run_directory / 'final_checkpoint.pt'}",
        f"--best-checkpoint={run_directory / 'best_checkpoint.pt'}",
        f"--seed={seed}",
        "--model=ViT-B-16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        "--epochs=20",
        "--batch-size=256",
        "--workers=4",
        "--lr=0.00001",
        "--wd=0.2",
        "--lr-scheduler=cosine",
        "--warmup-length=0",
        "--maple-precision=amp",
        "--best-metric=F1-macro_all",
        "--drm-weight=0",
        "--drm-warmup-epochs=0",
        "--tail-proto-weight=0",
        "--btel-weight=0",
        "--wise-alphas=0,0.05,0.1,0.15,0.2,0.3",
        "--no-wandb",
    ]


def execute_single_seed(seed: int, repository: str | Path, output_root: str | Path, source_commit: str) -> Path:
    repository = Path(repository).resolve()
    output_root = Path(output_root).resolve()
    if seed not in PILOT_SEEDS:
        raise ValueError(f"Seed {seed} is not in the preregistered pilot set.")
    if not (repository / "src" / "train_flyp.py").is_file():
        raise FileNotFoundError(f"Frozen source checkout is incomplete: {repository}")
    if "PAPER_GRADE_AMP_INIT_SCALE = 1.0" not in (
        repository / "src" / "train_flyp.py"
    ).read_text(encoding="utf-8"):
        raise RuntimeError("Frozen source checkout does not contain the repaired AMP policy.")

    run_directory = output_root / f"seed-{seed}"
    manifest_path = output_root / "single_seed_manifest.json"
    if run_directory.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to reuse single-seed output under {output_root}.")

    ensure_dependencies(repository)
    data_location = prepare_iwildcam_layout(repository)
    environment = dict(os.environ, PYTHONPATH=str(repository))
    run(
        build_command(seed, repository, output_root, data_location),
        cwd=repository,
        env=environment,
    )
    summary = verify_seed_run(run_directory, seed, source_commit)
    manifest = build_single_seed_manifest(summary)
    written = write_json_exclusive(manifest_path, manifest)
    print(json.dumps({"status": "complete", "seed": seed, "manifest": str(written)}, sort_keys=True))
    return written
