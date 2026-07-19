import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
OUTPUT_DIR = Path("/kaggle/working/lesvi-v0-val-confirm")
NEXT_LEDGER = Path("/kaggle/working/lesvi-v0-confirmation-ledger-next.json")
CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def find_file(environment_name, filename):
    configured = os.environ.get(environment_name)
    if configured and Path(configured).is_file():
        return Path(configured)
    matches = sorted(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Attach {filename} or set {environment_name}.")
    return matches[0]


def require_configured_file(environment_name):
    configured = os.environ.get(environment_name)
    if not configured or not Path(configured).is_file():
        raise FileNotFoundError(
            f"Set {environment_name} to the latest LESVI ledger explicitly; stale genesis auto-discovery is forbidden."
        )
    return Path(configured)


def find_data_location():
    configured = os.environ.get("LESVI_DATA_LOCATION")
    if configured and Path(configured).exists():
        return Path(configured)
    matches = sorted(Path("/kaggle/input").rglob("metadata.csv"))
    for metadata in matches:
        if "iwildcam" in str(metadata).lower():
            return metadata.parent.parent if metadata.parent.name == "iwildcam_v2.0" else metadata.parent
    raise FileNotFoundError("Attach IWildCam v2.0 or set LESVI_DATA_LOCATION.")


def configure_wandb_after_guard(environment):
    for name in WANDB_SECRET_NAMES:
        if environment.get(name):
            environment["WANDB_API_KEY"] = environment[name]
            return
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return
    client = UserSecretsClient()
    for name in WANDB_SECRET_NAMES:
        try:
            value = client.get_secret(name)
        except Exception:
            continue
        if value:
            environment["WANDB_API_KEY"] = value
            return


def main():
    clone_repository()
    run([
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
    ])
    if OUTPUT_DIR.exists():
        raise RuntimeError("LESVI final output already exists; refusing a repeated confirmation.")
    artifacts = {
        "frozen_spec": find_file("LESVI_FROZEN_SPEC", "lesvi_frozen_spec.json"),
        "ledger": require_configured_file("LESVI_CONFIRMATION_LEDGER"),
        "audit": find_file("LESVI_AUDIT_MANIFEST", "audit_manifest.json"),
        "viability": find_file("LESVI_VIABILITY_REPORT", "confirm_viability_aggregate.json"),
        "mapping": find_file("LESVI_CLASS_MAPPING", "class_mapping.json"),
        "prior": find_file("LESVI_PRIOR_ARTIFACT", "lesvi_prior.json"),
        "synthetic": find_file("LESVI_SYNTHETIC_VERIFICATION", "synthetic_verification.json"),
        "checkpoint": find_file("LESVI_CHECKPOINT", CHECKPOINT_NAME),
    }
    common = [
        sys.executable,
        "src/eval_lesvi_internal_confirm.py",
        f"--frozen-spec={artifacts['frozen_spec']}",
        f"--confirmation-ledger={artifacts['ledger']}",
        f"--next-ledger={NEXT_LEDGER}",
        f"--audit-manifest={artifacts['audit']}",
        f"--viability-report={artifacts['viability']}",
        f"--class-mapping={artifacts['mapping']}",
        f"--prior-artifact={artifacts['prior']}",
        f"--synthetic-verification={artifacts['synthetic']}",
        f"--checkpoint={artifacts['checkpoint']}",
        f"--data-location={find_data_location()}",
        f"--output-dir={OUTPUT_DIR}",
        f"--workspace-root={WORKING_REPOSITORY}",
        "--batch-size=256",
        "--workers=2",
        "--device=auto",
    ]
    environment = dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY))
    run([*common, "--validate-only"], cwd=WORKING_REPOSITORY, env=environment)
    configure_wandb_after_guard(environment)
    run(common, cwd=WORKING_REPOSITORY, env=environment)
    if not OUTPUT_DIR.is_dir() or not NEXT_LEDGER.is_file():
        raise RuntimeError("LESVI confirmation did not publish its atomic outputs.")


if __name__ == "__main__":
    main()
