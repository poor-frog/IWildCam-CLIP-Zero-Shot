import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
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


def ensure_repo_supports_selective_stp():
    evaluator = WORKING_REPOSITORY / "src" / "eval_tail_cache.py"
    adapter = WORKING_REPOSITORY / "src" / "models" / "stmp_adapter.py"
    legacy_checkpoint_module = WORKING_REPOSITORY / "src" / "models" / "modeling.py"
    if not evaluator.exists() or not adapter.exists() or not legacy_checkpoint_module.exists():
        raise RuntimeError("The cloned repo is incomplete. Push the selective STP implementation before rerunning this kernel.")
    evaluator_source = evaluator.read_text(encoding="utf-8")
    adapter_source = adapter.read_text(encoding="utf-8")
    legacy_source = legacy_checkpoint_module.read_text(encoding="utf-8")
    clip_model_source = (WORKING_REPOSITORY / "clip" / "model.py").read_text(encoding="utf-8")
    clip_source = (WORKING_REPOSITORY / "clip" / "clip.py").read_text(encoding="utf-8")
    if "--stp-selective-target" not in evaluator_source or "apply_target_selective_sequence_consensus" not in adapter_source:
        raise RuntimeError("The cloned repo is stale and does not contain the frozen selective STP implementation.")
    if "from .clip_encoder import" not in legacy_source or "VisualTransformer = VisionTransformer" not in clip_model_source or "_convert_to_rgb = _convert_image_to_rgb" not in clip_source:
        raise RuntimeError("The cloned repo is stale and cannot load the official FLYP pickle checkpoint.")


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


def patch_iwildcam_val():
    sys.path.insert(0, str(WORKING_REPOSITORY))
    import src.datasets as datasets
    import src.datasets.iwildcam as iwildcam

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


def find_checkpoint():
    candidates = [os.environ.get("FLYP_SELECTIVE_STP_CHECKPOINT")]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        candidates.extend(input_root.rglob(CHECKPOINT_NAME))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise FileNotFoundError(
        f"Could not find clean FLYP checkpoint {CHECKPOINT_NAME}. Attach a Kaggle dataset containing that exact file "
        "or set FLYP_SELECTIVE_STP_CHECKPOINT. This launcher will not substitute a DRM or WiSE checkpoint."
    )


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
        "--sequence-consensus-grid=0,0.5",
        "--multi-prototype-k-grid=1",
        "--multi-prototype-reduction=max",
        "--loo-bcpd-strength-grid=0",
        "--stp-selective-target",
        "--stp-selective-eta=0.5",
        "--stp-selective-margin-threshold=0.9341613054275513",
        "--stp-selective-min-burst-length=3",
        "--audit-metadata",
        "--max-cache-examples-per-class=0",
        "--batch-size=256",
        "--workers=2",
        "--device=auto",
        "--summary-head=stp_selective_target",
        "--selection-output=/kaggle/working/selective_stp_selection.json",
    ]
    if use_wandb:
        command.extend([
            "--wandb",
            "--wandb-project=PoorFrogs",
            "--wandb-run-name=flyp-selective-stp-v0-vitb16-iwildcamval",
        ])
    else:
        command.append("--no-wandb")
    return command


def main():
    clone_or_update_repository()
    ensure_dependencies()
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(WORKING_REPOSITORY), "--no-deps"])
    ensure_repo_supports_selective_stp()
    patch_iwildcam_val()
    checkpoint = find_checkpoint()
    data_location = prepare_iwildcam_layout()
    command = build_command(data_location, checkpoint, configure_wandb())
    run(command, cwd=WORKING_REPOSITORY, env=dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY)))


if __name__ == "__main__":
    main()
