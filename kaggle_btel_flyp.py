import ast
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_GITHUB_REPO = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
DEFAULT_REPO_ROOT = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
DATASET_CANDIDATES = (
    Path("/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"),
    Path("/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset"),
)
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")
REQUIRED_BTEL_FLAGS = (
    "--btel-weight",
    "--btel-prototype-scale",
    "--btel-negative-quantile",
    "--btel-max-frames-per-sequence",
)


class BTELLaunchError(RuntimeError):
    pass


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.check_call(command, cwd=cwd)


def ensure_repo() -> Path:
    if DEFAULT_REPO_ROOT.exists():
        run(["git", "-C", str(DEFAULT_REPO_ROOT), "pull", "--ff-only"])
        return DEFAULT_REPO_ROOT
    DEFAULT_REPO_ROOT.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", DEFAULT_GITHUB_REPO, str(DEFAULT_REPO_ROOT)])
    return DEFAULT_REPO_ROOT


def ensure_dependencies() -> None:
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "braceexpand",
            "ftfy",
            "numpy",
            "open-clip-torch",
            "pandas",
            "regex",
            "scikit-learn",
            "tqdm",
            "wandb",
            "webdataset",
            "wilds",
        ]
    )


def assert_btel_support(repo_root: Path) -> None:
    config_path = repo_root / "src" / "config.py"
    audit_path = repo_root / "src" / "audit_btel_sequences.py"
    train_path = repo_root / "src" / "train_flyp.py"
    runtime_paths = (repo_root / "src" / "models" / "btel.py", repo_root / "src" / "models" / "btel_artifacts.py")
    if not config_path.is_file() or not audit_path.is_file() or not train_path.is_file():
        raise BTELLaunchError("The cloned repository lacks BTEL training or audit files. Push the BTEL commit before rerunning.")
    missing_runtime_paths = [str(path.relative_to(repo_root)) for path in runtime_paths if not path.is_file()]
    if missing_runtime_paths:
        raise BTELLaunchError(f"The cloned repository lacks BTEL runtime files: {missing_runtime_paths}")
    source_paths = (config_path, audit_path, train_path, *runtime_paths)
    try:
        source_trees = {
            path: ast.parse(path.read_text(encoding="utf-8"), filename=str(path)) for path in source_paths
        }
    except SyntaxError as error:
        raise BTELLaunchError(f"The cloned repository contains invalid BTEL source: {error}") from error

    config_tree = source_trees[config_path]
    configured_flags = {
        argument.value
        for node in ast.walk(config_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
    }
    missing = [flag for flag in REQUIRED_BTEL_FLAGS if flag not in configured_flags]
    if missing:
        raise BTELLaunchError(f"The cloned repository is stale and lacks BTEL flags: {missing}")

    required_runtime_symbols = {
        runtime_paths[0]: {"BTELArtifacts", "BurstBatchSampler", "btel_sequence_loss"},
        runtime_paths[1]: {"audit_sequences", "build_btel_artifacts", "validate_btel_validation_split"},
    }
    for path, expected_symbols in required_runtime_symbols.items():
        definitions = {
            node.name
            for node in ast.walk(source_trees[path])
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        missing_symbols = sorted(expected_symbols - definitions)
        if missing_symbols:
            raise BTELLaunchError(f"The cloned repository has incomplete BTEL runtime code in {path.name}: {missing_symbols}")

    audit_definitions = {
        node.name for node in ast.walk(source_trees[audit_path]) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "main" not in audit_definitions:
        raise BTELLaunchError("The cloned repository has an incomplete BTEL sequence audit.")

    train_tree = source_trees[train_path]
    train_definitions = {
        node.name for node in ast.walk(train_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    train_source = ast.unparse(train_tree)
    if "maybe_prepare_btel" not in train_definitions or "btel_artifacts=" not in train_source:
        raise BTELLaunchError("The cloned repository is stale and does not wire BTEL into FLYP training.")

    smoke_code = (
        "import torch\n"
        "from src.models.btel import class_topk\n"
        "from src.models.btel_artifacts import find_empty_class_index\n"
        "assert class_topk(torch.tensor([20, 100, 101])).tolist() == [1, 2, 3]\n"
        "assert find_empty_class_index(['empty', 'animal']) == 0\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo_root)
    completed = subprocess.run(
        [sys.executable, "-c", smoke_code],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise BTELLaunchError("The cloned repository has incomplete BTEL runtime behavior.")


def find_iwildcam_root() -> Path:
    for root in DATASET_CANDIDATES:
        for candidate in (root, root / "iwildcam_v2.0", root / "archive", root / "archive" / "iwildcam_v2.0"):
            if (candidate / "metadata.csv").is_file():
                return candidate
    raise FileNotFoundError("Attach thanhquang71/iwildcam-v2-0-2020-wilds-dataset to this Kaggle kernel.")


def prepare_iwildcam_layout(repo_root: Path) -> str:
    source_root = find_iwildcam_root()
    target_root = repo_root / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)
    for source in source_root.iterdir():
        target = target_root / source.name
        if not target.exists() and not target.is_symlink():
            target.symlink_to(source, target_is_directory=source.is_dir())
    return str(target_root.parent)


def configure_wandb() -> bool:
    for name in WANDB_SECRET_NAMES:
        value = os.environ.get(name)
        if value:
            os.environ["WANDB_API_KEY"] = value
            return True
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False
    client = UserSecretsClient()
    for name in WANDB_SECRET_NAMES:
        try:
            value = client.get_secret(name)
        except Exception:  # noqa: BROAD_EXCEPT_OK - Kaggle exposes no stable missing-secret exception.
            continue
        if value:
            os.environ["WANDB_API_KEY"] = value
            print(f"Loaded W&B API key from Kaggle secret {name!r}.")
            return True
    return False


def main() -> None:
    repo_root = ensure_repo()
    ensure_dependencies()
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"])
    assert_btel_support(repo_root)
    data_location = prepare_iwildcam_layout(repo_root)
    batch_size = os.environ.get("BTEL_BATCH_SIZE", "256")
    workers = os.environ.get("BTEL_WORKERS", "2")
    epochs = os.environ.get("BTEL_EPOCHS", "20")
    weight = os.environ.get("BTEL_WEIGHT", "0.01")
    quantile = os.environ.get("BTEL_NEGATIVE_QUANTILE", "0.95")
    run_name = os.environ.get("BTEL_WANDB_RUN_NAME", "btel-flyp-lam0p01-q0p95-scale50-wise-vitb16-iwildcamval")
    common = [
        "--model=ViT-B-16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        f"--batch-size={batch_size}",
        f"--workers={workers}",
    ]
    run([sys.executable, "src/audit_btel_sequences.py", *common, "--no-wandb"], cwd=repo_root)
    command = [
        sys.executable,
        "src/train_flyp.py",
        *common,
        f"--epochs={epochs}",
        "--lr=1e-5",
        "--wd=0.2",
        "--lr-scheduler=cosine",
        "--maple-precision=amp",
        "--best-metric=F1-macro_all",
        "--drm-weight=0",
        "--tail-proto-weight=0",
        f"--btel-weight={weight}",
        "--btel-prototype-scale=50",
        f"--btel-negative-quantile={quantile}",
        "--btel-max-frames-per-sequence=8",
        "--wise-alphas=0.0,0.05,0.1,0.15,0.2,0.3",
        f"--save=/kaggle/working/checkpoints/btel_flyp_{weight.replace('.', 'p')}_q{quantile.replace('.', 'p')}_vitb16_iwildcamval.pt",
    ]
    if configure_wandb():
        command.extend(["--wandb", "--wandb-project=PoorFrogs", f"--wandb-run-name={run_name}"])
    else:
        command.append("--no-wandb")
    print("Running validation-selected FLYP + BTEL training:")
    run(command, cwd=repo_root)


if __name__ == "__main__":
    main()
