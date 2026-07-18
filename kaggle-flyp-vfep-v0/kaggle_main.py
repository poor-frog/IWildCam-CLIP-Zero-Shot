import os
import shutil
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
OUTPUT_NAME = "flyp-vfep-v0-val-audit"
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_repository():
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


def ensure_repo_supports_vfep():
    evaluator = WORKING_REPOSITORY / "src" / "eval_tail_cache.py"
    module = WORKING_REPOSITORY / "src" / "models" / "vfep.py"
    if not evaluator.is_file() or not module.is_file():
        raise RuntimeError("The cloned repository does not contain the VFEP v0 pilot implementation.")
    if "--vfep-pilot-output-dir" not in evaluator.read_text(encoding="utf-8"):
        raise RuntimeError("The cloned repository is stale. Push VFEP v0 before starting this kernel.")


def find_iwildcam_source_root():
    input_root = Path("/kaggle/input")
    candidates = [
        input_root / "iwildcam-v2-0-2020-wilds-dataset",
        input_root / "datasets" / "thanhquang71" / "iwildcam-v2-0-2020-wilds-dataset",
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
    candidates = [os.environ.get("FLYP_VFEP_CHECKPOINT")]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob(CHECKPOINT_NAME))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"Could not find {CHECKPOINT_NAME}. Attach the clean FLYP checkpoint dataset or set FLYP_VFEP_CHECKPOINT."
    )


def configure_wandb():
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
        "--loo-bcpd-strength-grid=0",
        "--vfep-strength-grid=0,0.25,0.5,1",
        "--vfep-stp-strength-grid=0,0.25,0.5,1",
        "--vfep-bootstrap-samples=2000",
        "--vfep-bootstrap-seed=20260718",
        "--vfep-shuffle-seed-start=20260718",
        "--vfep-shuffle-count=20",
        f"--vfep-pilot-output-dir=outputs/{OUTPUT_NAME}",
        "--audit-metadata",
        "--max-cache-examples-per-class=0",
        "--batch-size=256",
        "--workers=2",
        "--device=auto",
    ]
    if use_wandb:
        command.extend(["--wandb", "--wandb-project=PoorFrogs", "--wandb-run-name=flyp-vfep-v0-val-audit"])
    else:
        command.append("--no-wandb")
    return command


def publish_outputs():
    source = WORKING_REPOSITORY / "outputs" / OUTPUT_NAME
    if not source.is_dir():
        raise FileNotFoundError(f"Expected VFEP artifacts at {source}.")
    shutil.copytree(source, Path("/kaggle/working") / OUTPUT_NAME, dirs_exist_ok=True)


def main():
    clone_repository()
    ensure_dependencies()
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(WORKING_REPOSITORY), "--no-deps"])
    ensure_repo_supports_vfep()
    data_location = prepare_iwildcam_layout()
    checkpoint = find_checkpoint()
    environment = dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY))
    run(build_command(data_location, checkpoint, configure_wandb()), cwd=WORKING_REPOSITORY, env=environment)
    publish_outputs()


if __name__ == "__main__":
    main()
