import os
import subprocess
import sys
from pathlib import Path


DEFAULT_KAGGLE_DATASET = "/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"
DEFAULT_KAGGLE_DATASET_CANDIDATES = [
    DEFAULT_KAGGLE_DATASET,
    "/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset",
]
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
    "--val-dataset": "IWildCamVal",
    "--best-metric": "F1-macro_all",
    "--wandb-project": "PoorFrogs",
    "--wandb-run-name": "coop-vit-b32-phase11-best-f1",
    "--save": "./checkpoints/coop_prompt_learner.pt",
}
COOP_DEFAULT_FLAGS = ["--wandb"]

FULL_MAPLE_DEFAULTS = {
    "--model": "ViT-B/16",
    "--train-dataset": "IWildCam",
    "--eval-datasets": "IWildCamIDVal,IWildCamID,IWildCamOOD",
    "--batch-size": "256",
    "--workers": "4",
    "--n-ctx": "2",
    "--ctx-init": "a photo of a",
    "--epochs": "20",
    "--lr": "1e-5",
    "--wd": "0.2",
    "--lr-scheduler": "cosine",
    "--warmup-length": "500",
    "--maple-precision": "amp",
    "--template": "iwildcam_template",
    "--val-dataset": "IWildCamVal",
    "--best-metric": "F1-macro_all",
    "--maple-prompt-depth": "9",
    "--wandb-project": "PoorFrogs",
    "--wandb-run-name": "maple-full-vit-b16-bs256-iwildcamval",
    "--save": "./checkpoints/maple_full_prompt_learner_vitb16_bs256_iwildcamval.pt",
}
FULL_MAPLE_DEFAULT_FLAGS = ["--wandb"]

MAPLE_CBCE_DEFAULTS = {
    **FULL_MAPLE_DEFAULTS,
    "--eval-datasets": "IWildCamVal",
    "--val-dataset": "IWildCamVal",
    "--wandb-run-name": "a1-maple-cbce-vit-b16-bs256-iwildcamval",
    "--save": "/kaggle/working/checkpoints/a1_maple_cbce_vitb16_bs256_iwildcamval.pt",
}
MAPLE_CBCE_FLAGS = ["--wandb"]

MAPLE_TAU_SWEEP_EVAL_DEFAULTS = {
    **FULL_MAPLE_DEFAULTS,
    "--eval-datasets": "IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
    "--epochs": "0",
    "--selection-split": "IWildCamVal",
    "--logit-adjustment-tau-grid": "0,0.25,0.5,0.75,1,1.5,2",
    "--wandb-run-name": "maple-vanilla-vit-b16-bs256-tau-sweep-iwildcamval",
    "--load": "/kaggle/input/maple-vanilla-checkpoint/maple_full_prompt_learner_best.pt",
}
MAPLE_TAU_SWEEP_EVAL_FLAGS = ["--wandb"]

MAPLE_LORA_DEFAULTS = {
    **FULL_MAPLE_DEFAULTS,
    "--maple-lora-rank": "4",
    "--maple-lora-alpha": "8",
    "--maple-lora-layers": "last6",
    "--wandb-run-name": "maple-lora-vit-b16-bs256-r4-last6-e20-lr1e-5",
    "--save": "./checkpoints/maple_lora_vitb16_bs256_r4_last6_e20_lr1e-5.pt",
}
MAPLE_LORA_DEFAULT_FLAGS = ["--wandb"]

C1_DEFAULTS = {
    **MAPLE_LORA_DEFAULTS,
    "--eval-datasets": "IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
    "--val-dataset": "IWildCamVal",
    "--best-metric": "F1-macro_all",
    "--wandb-run-name": "c1-maple-lora-kl-vit-b16-bs256",
    "--save": "/kaggle/working/checkpoints/c1_maple_lora_kl_vitb16_bs256.pt",
}
C1_DEFAULT_FLAGS = ["--wandb"]

C1_AUTOFT_DEFAULTS = {
    **C1_DEFAULTS,
    "--val-dataset": "IWildCamOODVal",
    "--num-ood-hp-examples": "1000",
    "--wandb-run-name": "c1-autoft-1k-oodval-vit-b16-bs256",
    "--save": "/kaggle/working/checkpoints/c1_autoft_1k_oodval_vitb16_bs256.pt",
}
C1_AUTOFT_DEFAULT_FLAGS = ["--wandb", "--class-balanced-ood"]
C1_KL_WEIGHT = 0.1
C1_KL_TEMPERATURE = 1.0


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
    env_mode = os.environ.get("TRAIN_METHOD")
    if env_mode is not None:
        return env_mode
    try:
        from kaggle_secrets import UserSecretsClient
        secret_mode = UserSecretsClient().get_secret("TRAIN_METHOD")
        if secret_mode is not None:
            return secret_mode
    except Exception:
        pass
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


def build_full_maple_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**FULL_MAPLE_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in FULL_MAPLE_DEFAULT_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def build_maple_cbce_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**MAPLE_CBCE_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in MAPLE_CBCE_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def build_maple_tau_sweep_eval_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**MAPLE_TAU_SWEEP_EVAL_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in MAPLE_TAU_SWEEP_EVAL_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def build_maple_lora_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**MAPLE_LORA_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in MAPLE_LORA_DEFAULT_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def build_c1_autoft_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**C1_AUTOFT_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in C1_AUTOFT_DEFAULT_FLAGS:
        if flag not in provided and f"--no-{flag[2:]}" not in provided:
            argv.append(flag)

    argv.extend(user_args)
    return argv


def build_c1_training_argv(data_location, user_args=None):
    user_args = user_args or []
    argv = ["kaggle_main.py"]
    provided = _provided_option_names([argv[0], *user_args])

    defaults = {**C1_DEFAULTS, "--data-location": data_location}
    for name, value in defaults.items():
        if name not in provided:
            argv.append(f"{name}={value}")

    for flag in C1_DEFAULT_FLAGS:
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
        '"amp"',
    )
    missing = [fragment for fragment in required_fragments if fragment not in config_text]
    if missing:
        raise RuntimeError(
            "The cloned repo is stale and does not support the current Kaggle MaPLe/C1 flags "
            f"{missing}. Push the latest PoorFrogs code or set DEFAULT_GITHUB_REPO to a branch/commit "
            "that includes --maple-precision=amp, --lr-scheduler, and --warmup-length before rerunning."
        )


def _configure_wandb_from_kaggle_secret():
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False
    try:
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    except Exception:
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


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

    mode = parse_mode(sys.argv)
    if mode not in ("coop", "full_maple", "maple_cbce", "maple_tau_sweep", "maple_lora", "c1", "c1_autoft"):
        raise ValueError(
            f"Unknown mode: {mode}. Use --mode=coop, --mode=full_maple, --mode=maple_cbce, --mode=maple_tau_sweep, --mode=maple_lora, --mode=c1, or --mode=c1_autoft."
        )

    _ensure_deps()
    _ensure_local_package_installed(repo_root)
    assert_cloned_repo_supports_runtime_flags(repo_root)
    _patch_iwildcam_val()
    _configure_wandb_from_kaggle_secret()
    data_location = prepare_iwildcam_layout(repo_root)

    user_args = strip_mode_args(sys.argv)[1:]

    if mode == "full_maple":
        sys.argv = build_full_maple_training_argv(data_location, user_args)
        print("Running Kaggle full MaPLe training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_full import main as run_maple_full
        run_maple_full(parse_arguments())
    elif mode == "maple_cbce":
        sys.argv = build_maple_cbce_training_argv(data_location, user_args)
        print("Running Kaggle MaPLe + class-balanced CE training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_full import main as run_maple_full
        args = parse_arguments()
        args.class_balanced_ce = True
        run_maple_full(args)
    elif mode == "maple_tau_sweep":
        sys.argv = build_maple_tau_sweep_eval_argv(data_location, user_args)
        print("Running Kaggle vanilla MaPLe tau-sweep evaluation with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_full import main as run_maple_full
        run_maple_full(parse_arguments())
    elif mode == "maple_lora":
        sys.argv = build_maple_lora_training_argv(data_location, user_args)
        print("Running Kaggle MaPLe + LoRA training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_lora import configure_maple_lora_args
        from src.train_maple_full import main as run_maple_full
        run_maple_full(configure_maple_lora_args(parse_arguments()))
    elif mode == "c1_autoft":
        sys.argv = build_c1_autoft_training_argv(data_location, user_args)
        print("Running Kaggle C1 AutoFT-style 1k OODVal MaPLe + LoRA + KL training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_lora import configure_maple_lora_args
        from src.train_maple_full import main as run_maple_full
        args = configure_maple_lora_args(parse_arguments())
        args.kl_weight = float(os.environ.get("C1_KL_WEIGHT", C1_KL_WEIGHT))
        args.kl_temperature = float(os.environ.get("C1_KL_TEMPERATURE", C1_KL_TEMPERATURE))
        run_maple_full(args)
    elif mode == "c1":
        sys.argv = build_c1_training_argv(data_location, user_args)
        print("Running Kaggle C1 MaPLe + LoRA + KL training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_maple_lora import configure_maple_lora_args
        from src.train_maple_full import main as run_maple_full
        args = configure_maple_lora_args(parse_arguments())
        args.kl_weight = float(os.environ.get("C1_KL_WEIGHT", C1_KL_WEIGHT))
        args.kl_temperature = float(os.environ.get("C1_KL_TEMPERATURE", C1_KL_TEMPERATURE))
        run_maple_full(args)
    else:
        sys.argv = build_coop_training_argv(data_location, user_args)
        print("Running Kaggle CoOp training with arguments:")
        print(" ".join(sys.argv[1:]))
        from src.config import parse_arguments
        from src.train_coop import main as run_coop
        run_coop(parse_arguments())


if __name__ == "__main__":
    main()
