import os
import shutil
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
AUDIT_OUTPUT_NAME = "flyp-stp-mechanism-audit-v1-5"
HARDCODED_WANDB_API_KEY = ""
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_or_update_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def ensure_dependencies():
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


def ensure_repo_supports_audit():
    evaluator = WORKING_REPOSITORY / "src" / "eval_tail_cache.py"
    legacy_module = WORKING_REPOSITORY / "src" / "models" / "modeling.py"
    if not evaluator.exists() or not legacy_module.exists():
        raise RuntimeError("The cloned repo is incomplete. Push the clean-FLYP STP audit implementation before rerunning this kernel.")
    source = evaluator.read_text(encoding="utf-8")
    if "--stp-mechanism-audit-foundation" not in source or "stp_mechanism_audit_foundation" not in source:
        raise RuntimeError("The cloned repo is stale and does not support the clean-FLYP STP mechanism audit.")


def find_iwildcam_source_root():
    candidates = [
        Path("/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"),
        Path("/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset"),
    ]
    for root in candidates:
        if not root.exists():
            continue
        for candidate in (root, root / "iwildcam_v2.0", root / "archive" / "iwildcam_v2.0"):
            if (candidate / "metadata.csv").exists():
                return candidate
        matches = list(root.rglob("metadata.csv"))
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


def find_checkpoint():
    candidates = [os.environ.get("FLYP_STP_AUDIT_CHECKPOINT")]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob(CHECKPOINT_NAME))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"Could not find clean FLYP checkpoint {CHECKPOINT_NAME}. Attach a Kaggle dataset containing that exact file "
        "or set FLYP_STP_AUDIT_CHECKPOINT. This launcher will not substitute a DRM or WiSE checkpoint."
    )


def configure_wandb():
    if HARDCODED_WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
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


def build_command(data_location, checkpoint, use_wandb):
    command = [
        sys.executable,
        "src/eval_tail_cache.py",
        "--model=ViT-B-16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamVal",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        f"--load={checkpoint}",
        "--prototype-scale-grid=50",
        "--cache-tau-grid=0",
        "--tail-gamma-grid=0",
        "--gate-mode-grid=none",
        "--gate-strength-grid=0",
        "--sequence-consensus-grid=0",
        "--multi-prototype-k-grid=1",
        "--multi-prototype-reduction=max",
        "--loo-bcpd-strength-grid=0",
        "--stp-mechanism-audit-foundation=flyp",
        f"--stp-mechanism-audit-output-dir=outputs/{AUDIT_OUTPUT_NAME}",
        "--stp-mechanism-audit-bootstrap-samples=2000",
        "--audit-metadata",
        "--max-cache-examples-per-class=0",
        "--batch-size=256",
        "--workers=2",
        "--device=auto",
    ]
    if use_wandb:
        command.extend([
            "--wandb",
            "--wandb-project=PoorFrogs",
            "--wandb-run-name=flyp-stp-mechanism-audit-v1-5-val-audit",
        ])
    else:
        command.append("--no-wandb")
    return command


def publish_outputs():
    source = WORKING_REPOSITORY / "outputs" / AUDIT_OUTPUT_NAME
    if not source.is_dir():
        raise FileNotFoundError(f"Expected audit artifacts at {source}.")
    shutil.copytree(source, Path("/kaggle/working") / AUDIT_OUTPUT_NAME, dirs_exist_ok=True)


def main():
    clone_or_update_repository()
    ensure_dependencies()
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(WORKING_REPOSITORY), "--no-deps"])
    ensure_repo_supports_audit()
    checkpoint = find_checkpoint()
    data_location = prepare_iwildcam_layout()
    run(build_command(data_location, checkpoint, configure_wandb()), cwd=WORKING_REPOSITORY, env=dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY)))
    publish_outputs()


if __name__ == "__main__":
    main()
