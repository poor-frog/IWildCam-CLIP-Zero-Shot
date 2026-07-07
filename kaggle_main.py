import os
import subprocess
import sys
from pathlib import Path


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

DEFAULT_KAGGLE_DATASET = "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"
DEFAULT_KAGGLE_DATASET_CANDIDATES = [
    DEFAULT_KAGGLE_DATASET,
    "/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset",
]
DEFAULT_TAIL_TEACHER_CHECKPOINT = (
    "/kaggle/input/datasets/thanhquang71/flyp-nodrm-wise-vitb16-iwildcamval-checkpoint/"
    "flyp_nodrm_wise_vitb16_iwildcamval_best.pt"
)
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_GITHUB_REPO = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
DEFAULT_KAGGLE_WORKING_REPO = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
FLYP_WISE_FINE_ALPHAS = "0.0,0.05,0.1,0.15,0.2,0.3"
TAIL_AWARE_FLYP_WEIGHT = "0.003"
TAIL_AWARE_FLYP_SCALE = "50"
TAIL_AWARE_FLYP_OBJECTIVE = "fixed_distill"
TAIL_AWARE_FLYP_TEMPERATURE = "1.0"

FLYP_DEFAULTS = {
    "--model": "ViT-B-16",
    "--train-dataset": "IWildCam",
    "--eval-datasets": "IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
    "--batch-size": "128",
    "--workers": "2",
    "--epochs": "20",
    "--lr": "1e-5",
    "--wd": "0.2",
    "--lr-scheduler": "cosine",
    "--maple-precision": "amp",
    "--template": "iwildcam_template",
    "--val-dataset": "IWildCamVal",
    "--best-metric": "F1-macro_all",
    "--drm-weight": "0",
    "--drm-warmup-epochs": "0",
    "--tail-proto-weight": TAIL_AWARE_FLYP_WEIGHT,
    "--tail-proto-scale": TAIL_AWARE_FLYP_SCALE,
    "--tail-proto-objective": TAIL_AWARE_FLYP_OBJECTIVE,
    "--tail-proto-temperature": TAIL_AWARE_FLYP_TEMPERATURE,
    "--tail-proto-teacher-load": DEFAULT_TAIL_TEACHER_CHECKPOINT,
    "--wise-alphas": FLYP_WISE_FINE_ALPHAS,
    "--wandb-project": "PoorFrogs",
    "--wandb-run-name": "tail-aware-flyp-fixedtpa-distill-lam0p003-scale50-bs128-wise-vitb16-iwildcamval",
    "--save": "/kaggle/working/checkpoints/tail_aware_flyp_fixedtpa_distill_lam0p003_scale50_bs128_wise_vitb16_iwildcamval.pt",
}
FLYP_DEFAULT_FLAGS = ["--wandb"]

# ---------------------------------------------------------------------------
# Kernel overrides (Kaggle compatibility)
#
#   `environment_variables` in kernel-metadata.json is NOT supported by
#   Kaggle, so these hardcoded defaults let you create per-experiment
#   variants without requiring env vars or kernel-level secrets.  Set them
#   before pushing a new kernel, then `git commit` is not needed – the
#   filesystem copy you push is independent of the git working tree.
#
#   Each DRM-sweep variant should patch _KERNEL_DRM_WEIGHT only, and keep
#   the rest at None (which falls through to env var → CLI arg → built-in
#   default).
# ---------------------------------------------------------------------------
_KERNEL_DRM_WEIGHT = None              # None = env var → args.drm_weight
_KERNEL_WISE_ALPHAS = None             # None = env var → args.wise_alphas
_KERNEL_TAIL_PROTO_WEIGHT = None
_KERNEL_TAIL_PROTO_SCALE = None
_KERNEL_TAIL_PROTO_OBJECTIVE = None
_KERNEL_TAIL_PROTO_TEMPERATURE = None
_KERNEL_TAIL_PROTO_TEACHER_LOAD = None
_KERNEL_TAIL_PROTO_MAX_BATCHES = None
_KERNEL_WANDB_DISABLE = False          # True = never call wandb (no API key)
_KERNEL_WANDB_API_KEY = None
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def _keep_wandb_flag(default_flags):
    if _configure_wandb_from_kaggle_secret():
        return default_flags
    return [f for f in default_flags if f != "--wandb"]


def _configure_wandb_from_kaggle_secret():
    if _KERNEL_WANDB_DISABLE:
        return False
    for secret_name in WANDB_SECRET_NAMES:
        secret_value = os.environ.get(secret_name)
        if secret_value:
            os.environ["WANDB_API_KEY"] = secret_value
            return True
    if _KERNEL_WANDB_API_KEY:
        os.environ["WANDB_API_KEY"] = _KERNEL_WANDB_API_KEY
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

    print(f"WARNING: W&B logging is disabled. Tried Kaggle secrets: {', '.join(WANDB_SECRET_NAMES)}.", file=sys.stderr)
    return False

def _drm_weight_from_overrides():
    return _KERNEL_DRM_WEIGHT


def _wise_alphas_from_overrides():
    return _KERNEL_WISE_ALPHAS


def _tail_proto_weight_from_overrides():
    return _KERNEL_TAIL_PROTO_WEIGHT


def _tail_proto_scale_from_overrides():
    return _KERNEL_TAIL_PROTO_SCALE


def _tail_proto_objective_from_overrides():
    return _KERNEL_TAIL_PROTO_OBJECTIVE


def _tail_proto_temperature_from_overrides():
    return _KERNEL_TAIL_PROTO_TEMPERATURE


def _tail_proto_teacher_load_from_overrides():
    return _KERNEL_TAIL_PROTO_TEACHER_LOAD


def _tail_proto_max_batches_from_overrides():
    return _KERNEL_TAIL_PROTO_MAX_BATCHES


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


def _update_repo(repo_root, check_call=subprocess.check_call):
    """Pull latest from GitHub if the clone already exists."""
    try:
        check_call(["git", "-C", str(repo_root), "pull", "--ff-only"])
    except Exception:
        print("Warning: git pull --ff-only failed, continuing with existing clone.")


def ensure_repo_root(candidates=None, clone_target=DEFAULT_KAGGLE_WORKING_REPO, check_call=subprocess.check_call):
    candidates = candidates or [
        DEFAULT_REPO_ROOT,
        DEFAULT_KAGGLE_WORKING_REPO,
        Path("/kaggle/working/PoorFrogs"),
        Path("/kaggle/working"),
    ]
    repo_root = find_repo_root(candidates)
    if repo_root is not None:
        _update_repo(repo_root, check_call=check_call)
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


def resolve_iwildcam_source_root(kaggle_dataset_candidates=None):
    candidates = kaggle_dataset_candidates or DEFAULT_KAGGLE_DATASET_CANDIDATES
    missing_paths = []
    for candidate in candidates:
        try:
            return find_iwildcam_source_root(candidate)
        except FileNotFoundError:
            missing_paths.append(str(candidate))

    raise FileNotFoundError(
        "The IWildCam Kaggle dataset could not be found in any expected mount: "
        + ", ".join(missing_paths)
    )


def prepare_iwildcam_layout(repo_root, kaggle_dataset_path=DEFAULT_KAGGLE_DATASET, kaggle_dataset_candidates=None):
    repo_root = Path(repo_root)
    if kaggle_dataset_candidates is None and kaggle_dataset_path == DEFAULT_KAGGLE_DATASET:
        candidates = DEFAULT_KAGGLE_DATASET_CANDIDATES
    else:
        candidates = [kaggle_dataset_path, *(kaggle_dataset_candidates or [])]
    source_root = resolve_iwildcam_source_root(candidates)
    target_root = repo_root / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)

    links = {target_root / child.name: child for child in source_root.iterdir()}

    for link_path, source_path in links.items():
        if link_path.exists() or link_path.is_symlink() or not source_path.exists():
            continue
        link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())

    return str(target_root.parent)


def build_flyp_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**FLYP_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in _keep_wandb_flag(FLYP_DEFAULT_FLAGS):
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


def _ensure_local_package_installed(repo_root, check_call=subprocess.check_call):
    if not is_project_root(repo_root):
        return
    check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"])


def assert_cloned_repo_supports_runtime_flags(repo_root):
    config_path = Path(repo_root) / "src" / "config.py"
    if not config_path.exists():
        raise RuntimeError(f"The cloned repo is stale or incomplete: missing {config_path}.")
    config_text = config_path.read_text(encoding="utf-8")
    required_fragments = (
        "--lr-scheduler",
        "--warmup-length",
        "--drm-weight",
        "--drm-warmup-epochs",
        "--wise-alphas",
        "--wise-eval-alpha",
        "--tail-proto-weight",
        "--tail-proto-scale",
        "--tail-proto-objective",
        "--tail-proto-temperature",
        "--tail-proto-teacher-load",
        "--tail-proto-max-batches",
        '"amp"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in config_text]
    if missing:
        raise RuntimeError(
            "The cloned repo is stale and does not support the current Kaggle Tail-Aware FLYP flags "
            f"{missing}. Push the latest PoorFrogs code or set DEFAULT_GITHUB_REPO to a branch/commit "
            "that includes --maple-precision=amp, --lr-scheduler, FLYP DRM/WiSE flags, "
            "and prototype flags before rerunning."
        )


def _patch_iwildcam_val():
    """Patch IWildCamVal into cloned repo's datasets if missing."""
    try:
        import src.datasets as _ds
        import src.datasets.iwildcam as _iwildcam
    except ImportError:
        return  # dependencies may not be installed yet

    if hasattr(_ds, "IWildCamVal"):
        return

    class IWildCamVal(_iwildcam.IWildCam):
        def __init__(self, *args, **kwargs):
            kwargs["subset"] = "val"
            super().__init__(*args, **kwargs)

    _iwildcam.IWildCamVal = IWildCamVal
    _ds.IWildCamVal = IWildCamVal
    if hasattr(_ds, "__all__") and "IWildCamVal" not in _ds.__all__:
        _ds.__all__.append("IWildCamVal")
    print("Patched IWildCamVal into cloned repo datasets.")


def main():
    repo_root = ensure_repo_root()
    os.chdir(repo_root)
    configure_import_path(repo_root)

    _ensure_deps()
    _ensure_local_package_installed(repo_root)
    assert_cloned_repo_supports_runtime_flags(repo_root)
    _patch_iwildcam_val()
    _configure_wandb_from_kaggle_secret()
    data_location = prepare_iwildcam_layout(repo_root)

    user_args = strip_mode_args(sys.argv)[1:]

    sys.argv = build_flyp_training_argv(data_location, user_args)
    print("Running Kaggle Tail-Aware FLYP + WiSE training with arguments:")
    print(" ".join(sys.argv[1:]))
    from src.config import parse_arguments
    from src.train_flyp import main as run_flyp
    args = parse_arguments()
    drm_override = _drm_weight_from_overrides()
    if drm_override is not None:
        args.drm_weight = float(drm_override)
    else:
        args.drm_weight = float(os.environ.get("FLYP_DRM_WEIGHT", args.drm_weight))
    wise_override = _wise_alphas_from_overrides()
    if wise_override is not None:
        args.wise_alphas = wise_override
    else:
        args.wise_alphas = os.environ.get("FLYP_WISE_ALPHAS", args.wise_alphas)
    tail_weight_override = _tail_proto_weight_from_overrides()
    if tail_weight_override is not None:
        args.tail_proto_weight = float(tail_weight_override)
    else:
        args.tail_proto_weight = float(os.environ.get("FLYP_TAIL_PROTO_WEIGHT", args.tail_proto_weight))
    tail_scale_override = _tail_proto_scale_from_overrides()
    if tail_scale_override is not None:
        args.tail_proto_scale = float(tail_scale_override)
    else:
        args.tail_proto_scale = float(os.environ.get("FLYP_TAIL_PROTO_SCALE", args.tail_proto_scale))
    tail_objective_override = _tail_proto_objective_from_overrides()
    if tail_objective_override is not None:
        args.tail_proto_objective = tail_objective_override
    else:
        args.tail_proto_objective = os.environ.get("FLYP_TAIL_PROTO_OBJECTIVE", args.tail_proto_objective)
    tail_temperature_override = _tail_proto_temperature_from_overrides()
    if tail_temperature_override is not None:
        args.tail_proto_temperature = float(tail_temperature_override)
    else:
        args.tail_proto_temperature = float(os.environ.get("FLYP_TAIL_PROTO_TEMPERATURE", args.tail_proto_temperature))
    tail_teacher_load_override = _tail_proto_teacher_load_from_overrides()
    if tail_teacher_load_override is not None:
        args.tail_proto_teacher_load = tail_teacher_load_override
    else:
        args.tail_proto_teacher_load = os.environ.get("FLYP_TAIL_PROTO_TEACHER_LOAD", args.tail_proto_teacher_load)
    tail_max_batches_override = _tail_proto_max_batches_from_overrides()
    if tail_max_batches_override is not None:
        args.tail_proto_max_batches = int(tail_max_batches_override)
    elif os.environ.get("FLYP_TAIL_PROTO_MAX_BATCHES") is not None:
        args.tail_proto_max_batches = int(os.environ["FLYP_TAIL_PROTO_MAX_BATCHES"])
    run_flyp(args)


if __name__ == "__main__":
    main()
