import os
import subprocess
import sys
from pathlib import Path


DEFAULT_KAGGLE_DATASET = "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GITHUB_REPO = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
DEFAULT_KAGGLE_WORKING_REPO = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")

COOP_DEFAULTS = {
    "--model": "ViT-B/32",
    "--train-dataset": "IWildCam",
    "--eval-datasets": "IWildCamIDVal,IWildCamID,IWildCamOOD",
    "--batch-size": "32",
    "--workers": "4",
    "--n-ctx": "16",
    "--ctx-init": "a photo of a",
    "--epochs": "15",
    "--lr": "0.002",
    "--wd": "1e-5",
    "--val-dataset": "IWildCamIDVal",
    "--best-metric": "F1-macro_all",
    "--wandb-project": "PoorFrogs",
    "--wandb-run-name": "coop-vit-b32-phase11-best-f1",
    "--save": "./checkpoints/coop_prompt_learner.pt",
}
COOP_DEFAULT_FLAGS = ["--wandb"]


def strip_mode_args(argv):
    stripped = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--mode":
            skip_next = True
            continue
        if arg.startswith("--mode="):
            continue
        stripped.append(arg)
    return stripped


def parse_mode(argv):
    for index, arg in enumerate(argv):
        if arg == "--mode" and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith("--mode="):
            return arg.split("=", 1)[1]
    return "coop"


def _provided_option_names(argv):
    names = set()
    for arg in argv[1:]:
        if not arg.startswith("--"):
            continue
        names.add(arg.split("=", 1)[0])
    return names


def is_project_root(path):
    path = Path(path)
    has_package_config = (path / "pyproject.toml").exists() or (path / "setup.py").exists()
    return has_package_config and (path / "src" / "train_coop.py").exists()


def find_repo_root(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if is_project_root(candidate):
            return candidate
    return None


def ensure_repo_root(candidates=None, clone_target=DEFAULT_KAGGLE_WORKING_REPO, check_call=subprocess.check_call):
    candidates = candidates or [
        DEFAULT_REPO_ROOT,
        DEFAULT_KAGGLE_WORKING_REPO,
        Path("/kaggle/working/PoorFrogs"),
        Path("/kaggle/working"),
    ]
    repo_root = find_repo_root(candidates)
    if repo_root is not None:
        return repo_root

    clone_target = Path(clone_target)
    if not clone_target.exists():
        clone_target.parent.mkdir(parents=True, exist_ok=True)
        check_call(["git", "clone", DEFAULT_GITHUB_REPO, str(clone_target)])

    repo_root = find_repo_root([clone_target])
    if repo_root is None:
        raise FileNotFoundError(f"Could not locate or clone PoorFrogs repo at {clone_target}")
    return repo_root


def configure_import_path(repo_root):
    repo_root = str(repo_root)
    os.environ["PYTHONPATH"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def resolve_kaggle_data_location(repo_root, fallback_input_path=DEFAULT_KAGGLE_DATASET):
    local_data = Path(repo_root) / "data" / "iwildcam_v2.0"
    if local_data.exists():
        return str(local_data.parent)
    return fallback_input_path


def prepare_iwildcam_layout(repo_root, kaggle_dataset_path=DEFAULT_KAGGLE_DATASET):
    repo_root = Path(repo_root)
    source_root = Path(kaggle_dataset_path)
    target_root = repo_root / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)

    archive = source_root / "archive"
    if archive.exists():
        links = {
            target_root / "archive": archive,
            target_root / "train": archive / "train",
            target_root / "metadata.csv": archive / "metadata.csv",
        }
    else:
        links = {
            target_root / "train": source_root / "train",
            target_root / "metadata.csv": source_root / "metadata.csv",
        }

    for link_path, source_path in links.items():
        if link_path.exists() or link_path.is_symlink() or not source_path.exists():
            continue
        link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())

    return str(target_root.parent)


def build_coop_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**COOP_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in COOP_DEFAULT_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def _ensure_deps():
    packages = [
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _ensure_local_package_installed(repo_root):
    if not is_project_root(repo_root):
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root)])


def _configure_wandb_from_kaggle_secret():
    if os.environ.get("WANDB_API_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return
    try:
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        return


def main():
    repo_root = ensure_repo_root()
    os.chdir(repo_root)
    configure_import_path(repo_root)

    mode = parse_mode(sys.argv)
    if mode != "coop":
        raise ValueError("This Kaggle entrypoint is for CoOp training. Use --mode=coop or omit --mode.")

    _ensure_deps()
    _ensure_local_package_installed(repo_root)
    _configure_wandb_from_kaggle_secret()
    data_location = prepare_iwildcam_layout(repo_root)

    user_args = strip_mode_args(sys.argv)[1:]
    sys.argv = build_coop_training_argv(data_location, user_args)

    print("Running Kaggle CoOp training with arguments:")
    print(" ".join(sys.argv[1:]))

    from src.config import parse_arguments
    from src.train_coop import main as run_coop

    run_coop(parse_arguments())


if __name__ == "__main__":
    main()
