import os
import subprocess
import sys
from pathlib import Path


DEFAULT_GITHUB_REPO = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
DEFAULT_DRM_GITHUB_REPO = "https://github.com/vaynexie/DRM.git"
DEFAULT_KAGGLE_WORKING_REPO = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
DEFAULT_DRM_REPO = Path("/kaggle/working/DRM")
DEFAULT_IWILDCAM_CANDIDATES = (
    Path("/kaggle/input/iwildcam-v2-0-2020-wilds-dataset"),
    Path("/kaggle/input/datasets/thanhquang71/iwildcam-v2-0-2020-wilds-dataset"),
)
DRM_CHECKPOINT_NAME = os.environ.get("DRM_CHECKPOINT_NAME", "iwildcam_vit_b16.pt")
PROTOTYPE_SCALE_GRID = os.environ.get("DRM_STMP_PROTOTYPE_SCALE_GRID", "50")
CONCEPT_BETA_GRID = os.environ.get("DRM_STMP_CONCEPT_BETA_GRID", "0.5")
TAIL_GAMMA_GRID = os.environ.get("DRM_STMP_TAIL_GAMMA_GRID", "0")
GATE_MODE_GRID = os.environ.get("DRM_STMP_GATE_MODE_GRID", "none")
GATE_STRENGTH_GRID = os.environ.get("DRM_STMP_GATE_STRENGTH_GRID", "0")
SEQUENCE_CONSENSUS_GRID = os.environ.get("DRM_STMP_SEQUENCE_CONSENSUS_GRID", "0,0.25,0.5")
MULTI_PROTOTYPE_K_GRID = os.environ.get("DRM_STMP_MULTI_PROTOTYPE_K_GRID", "1,2,4,8")
MULTI_PROTOTYPE_REDUCTION = os.environ.get("DRM_STMP_MULTI_PROTOTYPE_REDUCTION", "max")
BATCH_SIZE = os.environ.get("DRM_STMP_BATCH_SIZE", "256")
WORKERS = os.environ.get("DRM_STMP_WORKERS", "2")
WANDB_RUN_NAME = os.environ.get(
    "DRM_STMP_WANDB_RUN_NAME",
    "drm-stmp-multiprototype-k1-2-4-8-seq0-0p25-0p5-vitb16-iwildcamval",
)


class DrmConceptEvalSupportError(RuntimeError):
    pass

# Optional private-kernel fallback when Kaggle secrets are unavailable.
# Paste your W&B key here before `kaggle kernels push -p .`, then clear it before
# any GitHub commit/push.
HARDCODED_WANDB_API_KEY = ""
WANDB_SECRET_NAMES = ("WANDB_API_KEY", "wandb-api-key", "wandb_api_key", "WANDB-API-KEY")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def ensure_deps():
    packages = [
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
    run([sys.executable, "-m", "pip", "install", "-q", *packages])


def clone_or_update(repo_url, target):
    if target.exists():
        run(["git", "-C", target, "pull", "--ff-only"])
    else:
        run(["git", "clone", repo_url, target])
    return target


def configure_import_path(repo_root):
    repo_root = str(repo_root)
    os.environ["PYTHONPATH"] = repo_root
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def ensure_local_package_installed(repo_root):
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_root), "--no-deps"])


def patch_tail_cache_eval_guard(repo_root):
    path = Path(repo_root) / "src" / "eval_tail_cache.py"
    text = path.read_text()
    text = text.replace("from src.train_flyp import clone_state_dict, ensure_open_clip_for_flyp", "from src.train_flyp import clone_state_dict")
    text = text.replace("    ensure_open_clip_for_flyp(args.model)\n\n", "")
    text = text.replace("--load must point to a FLYP CLIPEncoder checkpoint.", "--load must point to a CLIPEncoder checkpoint.")
    path.write_text(text)


def assert_repo_supports_drm_concept_eval(repo_root):
    helper_path = Path(repo_root) / "src" / "eval_drm_blend.py"
    if not helper_path.is_file():
        raise DrmConceptEvalSupportError(
            "The cloned repo lacks src/eval_drm_blend.py required by --cd-path. "
            "Push the DRM concept-evaluation helper to origin/main before rerunning."
        )


def find_iwildcam_source_root():
    for root in DEFAULT_IWILDCAM_CANDIDATES:
        for candidate in (root, root / "iwildcam_v2.0", root / "archive", root / "archive" / "iwildcam_v2.0"):
            if (candidate / "metadata.csv").exists():
                return candidate
    for metadata_path in Path("/kaggle/input").rglob("metadata.csv"):
        if metadata_path.parent.name == "iwildcam_v2.0":
            return metadata_path.parent
    raise FileNotFoundError("Could not find iwildcam_v2.0/metadata.csv under /kaggle/input.")


def prepare_iwildcam_layout(repo_root):
    source_root = find_iwildcam_source_root()
    target_root = Path(repo_root) / "data" / "iwildcam_v2.0"
    target_root.mkdir(parents=True, exist_ok=True)
    for source_path in source_root.iterdir():
        link_path = target_root / source_path.name
        if link_path.exists() or link_path.is_symlink():
            continue
        link_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    return str(target_root.parent)


def find_drm_checkpoint():
    env_path = os.environ.get("DRM_CHECKPOINT_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    matches = sorted(Path("/kaggle/input").rglob(DRM_CHECKPOINT_NAME))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {DRM_CHECKPOINT_NAME} under /kaggle/input.")


def export_drm_state_dict(drm_repo, checkpoint_path, state_dict_path):
    code = "\n".join([
        "import sys",
        "import torch",
        "from src.models import utils",
        "checkpoint_path, state_dict_path = sys.argv[1:3]",
        "obj = utils.torch_load(checkpoint_path)",
        'model = obj.model if hasattr(obj, "model") else obj',
        "torch.save(model.state_dict(), state_dict_path)",
        'print(f"Exported DRM model state_dict to {state_dict_path}")',
    ])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(drm_repo)
    run([sys.executable, "-c", code, str(checkpoint_path), str(state_dict_path)], cwd=drm_repo, env=env)


def build_poorfrogs_checkpoint(repo_root, state_dict_path, output_path):
    code = "\n".join([
        "import sys",
        "import torch",
        "from types import SimpleNamespace",
        "from src.models.clip_encoder import CLIPEncoder",
        "state_dict_path, output_path = sys.argv[1:3]",
        'args = SimpleNamespace(model="ViT-B/16", device="cpu", cache_dir=None)',
        "encoder = CLIPEncoder(args, keep_lang=True)",
        'state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=False)',
        "encoder.model.load_state_dict(state_dict, strict=True)",
        "encoder.save(output_path)",
        'print(f"Saved PoorFrogs-compatible CLIPEncoder checkpoint to {output_path}")',
    ])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root)
    run([sys.executable, "-c", code, str(state_dict_path), str(output_path)], cwd=repo_root, env=env)


def patch_iwildcam_val():
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
        except Exception:  # noqa: BROAD_EXCEPT_OK - Kaggle secret lookup has no stable missing-secret exception.
            continue
        if secret_value:
            os.environ["WANDB_API_KEY"] = secret_value
            return True
    return False


def main():
    ensure_deps()
    repo_root = clone_or_update(DEFAULT_GITHUB_REPO, DEFAULT_KAGGLE_WORKING_REPO)
    drm_repo = clone_or_update(DEFAULT_DRM_GITHUB_REPO, DEFAULT_DRM_REPO)
    configure_import_path(repo_root)
    assert_repo_supports_drm_concept_eval(repo_root)
    ensure_local_package_installed(repo_root)
    patch_tail_cache_eval_guard(repo_root)
    patch_iwildcam_val()

    data_location = prepare_iwildcam_layout(repo_root)
    drm_checkpoint = find_drm_checkpoint()
    concept_path = drm_repo / "prompts" / "iwildcam_cd.json"
    state_dict_path = Path("/kaggle/working/drm_iwildcam_vit_b16_state_dict.pt")
    converted_checkpoint = Path("/kaggle/working/drm_iwildcam_vit_b16_poorfrogs_clip_encoder.pt")
    export_drm_state_dict(drm_repo, drm_checkpoint, state_dict_path)
    build_poorfrogs_checkpoint(repo_root, state_dict_path, converted_checkpoint)

    command = [
        sys.executable,
        "src/eval_tail_cache.py",
        "--model=ViT-B/16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_drm_template",
        f"--data-location={data_location}",
        f"--load={converted_checkpoint}",
        f"--cd-path={concept_path}",
        f"--concept-beta-grid={CONCEPT_BETA_GRID}",
        f"--prototype-scale-grid={PROTOTYPE_SCALE_GRID}",
        "--cache-tau-grid=0",
        f"--tail-gamma-grid={TAIL_GAMMA_GRID}",
        f"--gate-mode-grid={GATE_MODE_GRID}",
        f"--gate-strength-grid={GATE_STRENGTH_GRID}",
        f"--sequence-consensus-grid={SEQUENCE_CONSENSUS_GRID}",
        "--sequence-id-field=auto",
        f"--multi-prototype-k-grid={MULTI_PROTOTYPE_K_GRID}",
        f"--multi-prototype-reduction={MULTI_PROTOTYPE_REDUCTION}",
        "--audit-metadata",
        "--report-key-ablation-candidates",
        "--max-cache-examples-per-class=0",
        f"--batch-size={BATCH_SIZE}",
        f"--workers={WORKERS}",
        "--device=auto",
    ]
    if configure_wandb():
        command.extend([
            "--wandb",
            "--wandb-project=PoorFrogs",
            f"--wandb-run-name={WANDB_RUN_NAME}",
        ])
    else:
        command.append("--no-wandb")
    print("Running official DRM concept-description parity evaluation:")
    print(" ".join(str(part) for part in command))
    run(command, cwd=repo_root)


if __name__ == "__main__":
    main()
