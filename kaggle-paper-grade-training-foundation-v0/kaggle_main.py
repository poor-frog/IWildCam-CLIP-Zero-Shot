import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
SOURCE_COMMIT = "4246a27c13db27b832a7441b9ae31a880bcdd8f6"
WORKING_REPOSITORY = Path("/kaggle/working/pgf-v0-source")
OUTPUT_ROOT = Path("/kaggle/working/paper-grade-training-foundation-v0-pilot")
PILOT_SEEDS = (20260721, 20260722, 20260723)
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_frozen_repository():
    if WORKING_REPOSITORY.exists():
        current_commit = subprocess.check_output(
            ["git", "-C", str(WORKING_REPOSITORY), "rev-parse", "HEAD"],
            text=True,
        ).strip()
        if current_commit != SOURCE_COMMIT:
            raise RuntimeError(
                f"Existing source checkout is {current_commit}; frozen PGF-03 source is {SOURCE_COMMIT}."
            )
        return
    run(["git", "clone", "--no-checkout", GITHUB_REPOSITORY, WORKING_REPOSITORY])
    run(["git", "-C", WORKING_REPOSITORY, "checkout", "--detach", SOURCE_COMMIT])


def ensure_dependencies():
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "braceexpand",
            "ftfy",
            "open-clip-torch",
            "pandas",
            "regex",
            "tqdm",
            "wandb",
            "webdataset",
            "wilds",
        ]
    )
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(WORKING_REPOSITORY), "--no-deps"])


def ensure_frozen_source_support():
    required = {
        WORKING_REPOSITORY / "src" / "train_flyp.py": "paper_grade_training_foundation_v0",
        WORKING_REPOSITORY / "src" / "models" / "paper_grade_training_foundation.py": "ValidationSplitFirewall",
        WORKING_REPOSITORY / "experiments" / "paper_grade_training_foundation_v0" / "preregistration.json": SOURCE_COMMIT,
    }
    for path, marker in required.items():
        if not path.is_file():
            raise RuntimeError(f"Frozen source checkout is missing {path.relative_to(WORKING_REPOSITORY)}")
        if path.suffix == ".py" and marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(f"Frozen source checkout lacks PGF-02 marker {marker!r} in {path.name}")


def find_iwildcam_source_root():
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


def prepare_iwildcam_layout():
    source_root = find_iwildcam_source_root()
    target_root = WORKING_REPOSITORY / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in source_root.iterdir():
        target_path = target_root / source_path.name
        if not target_path.exists() and not target_path.is_symlink():
            target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    return target_root.parent


def configure_wandb():
    if os.environ.get("WANDB_API_KEY"):
        return True
    for name in WANDB_SECRET_NAMES:
        if os.environ.get(name):
            os.environ["WANDB_API_KEY"] = os.environ[name]
            return True
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False
    client = UserSecretsClient()
    for name in WANDB_SECRET_NAMES:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            os.environ["WANDB_API_KEY"] = value
            return True
    return False


def seed_run_directory(seed):
    return OUTPUT_ROOT / f"seed-{seed}"


def build_command(seed, data_location, use_wandb):
    run_directory = seed_run_directory(seed)
    command = [
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
    ]
    if use_wandb:
        command.extend(
            [
                "--wandb",
                "--wandb-project=PoorFrogs",
                f"--wandb-run-name=pgf-v0-pilot-seed-{seed}",
            ]
        )
    else:
        command.append("--no-wandb")
    return command


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_seed_receipt(seed):
    receipt_path = seed_run_directory(seed) / "run_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("seed") != seed:
        raise RuntimeError(f"Seed {seed} receipt is incomplete or has the wrong seed.")
    if receipt.get("provenance", {}).get("source", {}).get("git_commit") != SOURCE_COMMIT:
        raise RuntimeError(f"Seed {seed} receipt is not bound to frozen source {SOURCE_COMMIT}.")
    if receipt.get("split_firewall", {}).get("accessed_datasets") != ["IWildCam", "IWildCamVal"]:
        raise RuntimeError(f"Seed {seed} accessed data outside train plus IWildCamVal.")
    if not all(receipt.get("validity", {}).values()):
        raise RuntimeError(f"Seed {seed} receipt failed its provenance or firewall validity checks.")
    for artifact in receipt["provenance"]["checkpoints"].values():
        if sha256_file(artifact["path"]) != artifact["sha256"]:
            raise RuntimeError(f"Seed {seed} checkpoint hash mismatch: {artifact['path']}")
    return {
        "seed": seed,
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "source_tree_sha256": receipt["provenance"]["source"]["source_tree_sha256"],
        "dataset_snapshot_sha256": receipt["provenance"]["dataset"]["dataset_snapshot_sha256"],
        "protocol_configuration_sha256": receipt["protocol_configuration_sha256"],
        "best_checkpoint_sha256": receipt["provenance"]["checkpoints"]["best_validation"]["sha256"],
        "final_checkpoint_sha256": receipt["provenance"]["checkpoints"]["final_wise"]["sha256"],
        "best_epoch": receipt["training"]["best_epoch"],
        "selected_wise_alpha": receipt["training"]["selected_wise_alpha"],
    }


def build_pilot_manifest(seed_receipts):
    if [item["seed"] for item in seed_receipts] != list(PILOT_SEEDS):
        raise RuntimeError("Pilot manifest must contain the three preregistered seeds in order.")
    singleton_fields = (
        "source_tree_sha256",
        "dataset_snapshot_sha256",
        "protocol_configuration_sha256",
    )
    for field in singleton_fields:
        if len({item[field] for item in seed_receipts}) != 1:
            raise RuntimeError(f"Pilot seeds disagree on {field}.")
    for field in ("best_checkpoint_sha256", "final_checkpoint_sha256"):
        if len({item[field] for item in seed_receipts}) != len(PILOT_SEEDS):
            raise RuntimeError(f"Pilot seed checkpoints are not unique for {field}.")
    return {
        "manifest": "paper_grade_training_foundation_v0_three_seed_pilot",
        "status": "complete",
        "source_commit": SOURCE_COMMIT,
        "seeds": list(PILOT_SEEDS),
        "selection_split": "IWildCamVal",
        "final_splits_opened": False,
        "seed_receipts": seed_receipts,
    }


def write_pilot_manifest(seed_receipts):
    manifest_path = OUTPUT_ROOT / "pilot_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(build_pilot_manifest(seed_receipts), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote immutable pilot manifest to {manifest_path}")


def main():
    clone_frozen_repository()
    ensure_frozen_source_support()
    ensure_dependencies()
    data_location = prepare_iwildcam_layout()
    use_wandb = configure_wandb()
    environment = dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY))
    seed_receipts = []
    for seed in PILOT_SEEDS:
        run_directory = seed_run_directory(seed)
        if run_directory.exists():
            raise FileExistsError(f"Refusing to reuse pilot seed directory: {run_directory}")
        run(build_command(seed, data_location, use_wandb), cwd=WORKING_REPOSITORY, env=environment)
        seed_receipts.append(verify_seed_receipt(seed))
    write_pilot_manifest(seed_receipts)


if __name__ == "__main__":
    main()
