import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_KAGGLE_DATASET = "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"
DEFAULT_KAGGLE_DATASET_CANDIDATES = [
    DEFAULT_KAGGLE_DATASET,
    "/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset",
]
DEFAULT_GITHUB_REPO = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
DEFAULT_KAGGLE_WORKING_REPO = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
HARDCODED_WANDB_API_KEY = ""
WANDB_SECRET_NAMES = (
    "WANDB_API_KEY",
    "wandb-api-key",
    "wandb_api_key",
    "WANDB-API-KEY",
)


def is_project_root(path):
    path = Path(path)
    has_package_config = (path / "pyproject.toml").exists() or (path / "setup.py").exists()
    return has_package_config and (path / "src" / "train_flyp.py").exists()


def find_repo_root(candidates):
    for candidate in candidates:
        candidate = Path(candidate)
        if is_project_root(candidate):
            return candidate
    return None


def update_repo(repo_root):
    try:
        subprocess.check_call(["git", "-C", str(repo_root), "pull", "--ff-only"])
    except Exception:
        print("Warning: git pull --ff-only failed, continuing with existing clone.")


def ensure_repo_root():
    candidates = [
        Path(__file__).resolve().parent,
        DEFAULT_KAGGLE_WORKING_REPO,
        Path("/kaggle/working/PoorFrogs"),
        Path("/kaggle/working"),
    ]
    repo_root = find_repo_root(candidates)
    if repo_root is not None:
        update_repo(repo_root)
        return repo_root

    DEFAULT_KAGGLE_WORKING_REPO.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_KAGGLE_WORKING_REPO.exists():
        subprocess.check_call(["git", "clone", DEFAULT_GITHUB_REPO, str(DEFAULT_KAGGLE_WORKING_REPO)])

    repo_root = find_repo_root([DEFAULT_KAGGLE_WORKING_REPO])
    if repo_root is None:
        raise FileNotFoundError(f"Could not locate or clone PoorFrogs repo at {DEFAULT_KAGGLE_WORKING_REPO}")
    return repo_root


def configure_import_path(repo_root):
    repo_root = str(repo_root)
    os.environ["PYTHONPATH"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_deps():
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


def ensure_local_package_installed(repo_root):
    if is_project_root(repo_root):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"])


def find_iwildcam_source_root(kaggle_dataset_path):
    source_root = Path(kaggle_dataset_path)
    if not source_root.exists():
        raise FileNotFoundError(f"IWildCam Kaggle dataset path does not exist: {source_root}")
    candidates = [
        source_root,
        source_root / "iwildcam_v2.0",
        source_root / "archive",
        source_root / "archive" / "iwildcam_v2.0",
    ]
    for candidate in candidates:
        if (candidate / "metadata.csv").exists():
            return candidate

    for metadata_path in source_root.rglob("metadata.csv"):
        if metadata_path.parent.name == "iwildcam_v2.0":
            return metadata_path.parent

    return source_root


def resolve_iwildcam_source_root():
    missing_paths = []
    for candidate in DEFAULT_KAGGLE_DATASET_CANDIDATES:
        try:
            return find_iwildcam_source_root(candidate)
        except FileNotFoundError:
            missing_paths.append(candidate)
    raise FileNotFoundError(
        "The IWildCam Kaggle dataset could not be found in any expected mount: "
        + ", ".join(missing_paths)
    )


def prepare_iwildcam_layout(repo_root):
    repo_root = Path(repo_root)
    source_root = resolve_iwildcam_source_root()
    target_root = repo_root / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)

    for source_path in source_root.iterdir():
        link_path = target_root / source_path.name
        if link_path.exists() or link_path.is_symlink() or not source_path.exists():
            continue
        link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())

    return str(target_root.parent)


def patch_iwildcam_val():
    try:
        import src.datasets as datasets
        import src.datasets.iwildcam as iwildcam
    except ImportError:
        return

    if hasattr(datasets, "IWildCamVal"):
        return

    class IWildCamVal(iwildcam.IWildCam):
        def __init__(self, *args, **kwargs):
            kwargs["subset"] = "val"
            super().__init__(*args, **kwargs)

    iwildcam.IWildCamVal = IWildCamVal
    datasets.IWildCamVal = IWildCamVal
    if hasattr(datasets, "__all__") and "IWildCamVal" not in datasets.__all__:
        datasets.__all__.append("IWildCamVal")
    print("Patched IWildCamVal into cloned repo datasets.")


def configure_wandb():
    if HARDCODED_WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
        return True

    for secret_name in WANDB_SECRET_NAMES:
        secret_value = os.environ.get(secret_name)
        if secret_value:
            os.environ["WANDB_API_KEY"] = secret_value
            return True

    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False

    secrets_client = UserSecretsClient()
    for secret_name in WANDB_SECRET_NAMES:
        try:
            secret_value = secrets_client.get_secret(secret_name)
        except Exception:
            continue
        if secret_value:
            os.environ["WANDB_API_KEY"] = secret_value
            print(f"Loaded W&B API key from Kaggle secret {secret_name!r}.")
            return True

    print(
        "W&B logging is disabled because no Kaggle secret was found. "
        f"Tried: {', '.join(WANDB_SECRET_NAMES)}.",
        file=sys.stderr,
    )
    return bool(os.environ.get("WANDB_API_KEY"))


def main():
    repo_root = ensure_repo_root()
    os.chdir(repo_root)
    configure_import_path(repo_root)
    ensure_deps()
    ensure_local_package_installed(repo_root)
    patch_iwildcam_val()

    data_location = prepare_iwildcam_layout(repo_root)
    batch_size = os.environ.get("FLYP_VITL14_BATCH_SIZE", "16")
    workers = os.environ.get("FLYP_VITL14_WORKERS", "2")
    epochs = os.environ.get("FLYP_VITL14_EPOCHS", "20")
    save_path = "/kaggle/working/checkpoints/flyp_nodrm_wise_vitl14_bs16_iwildcamval.pt"
    command = [
        sys.executable,
        "src/train_flyp.py",
        "--model=ViT-L-14",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        f"--batch-size={batch_size}",
        f"--workers={workers}",
        f"--epochs={epochs}",
        "--lr=1e-5",
        "--wd=0.2",
        "--lr-scheduler=cosine",
        "--maple-precision=amp",
        "--best-metric=F1-macro_all",
        "--drm-weight=0",
        "--drm-warmup-epochs=0",
        "--wise-alphas=0.0,0.05,0.1,0.15,0.2,0.3",
        f"--save={save_path}",
        "--device=auto",
    ]
    if configure_wandb():
        command.extend([
            "--wandb",
            "--wandb-project=PoorFrogs",
            "--wandb-run-name=flyp-nodrm-wise-vitl14-bs16-iwildcamval",
        ])
    else:
        command.append("--no-wandb")

    print("Running Kaggle FLYP no-DRM WiSE ViT-L/14 training:")
    print(" ".join(command))
    subprocess.check_call(command)


if __name__ == "__main__":
    main()
