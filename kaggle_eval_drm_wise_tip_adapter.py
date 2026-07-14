import os
import sys
from pathlib import Path

from kaggle_eval_drm_stmp_adapter import (
    DEFAULT_DRM_GITHUB_REPO,
    DEFAULT_DRM_REPO,
    DEFAULT_GITHUB_REPO,
    DEFAULT_KAGGLE_WORKING_REPO,
    build_poorfrogs_checkpoint,
    clone_or_update,
    configure_import_path,
    configure_wandb,
    ensure_deps,
    ensure_local_package_installed,
    export_drm_state_dict,
    find_drm_checkpoint,
    patch_iwildcam_val,
    patch_tail_cache_eval_guard,
    prepare_iwildcam_layout,
    run,
)


FIXED_WISE_ALPHA = "0.2"
TIP_ADAPTER_BETA_GRID = os.environ.get("DRM_TIP_ADAPTER_BETA_GRID", "0.1,0.5,1,2,5,7")
TIP_ADAPTER_ALPHA_GRID = os.environ.get("DRM_TIP_ADAPTER_ALPHA_GRID", "0.1,0.5,1,2,3")
TIP_ADAPTER_QUERY_CHUNK_SIZE = os.environ.get("DRM_TIP_ADAPTER_QUERY_CHUNK_SIZE", "256")
TIP_ADAPTER_CACHE_CHUNK_SIZE = os.environ.get("DRM_TIP_ADAPTER_CACHE_CHUNK_SIZE", "16384")
BATCH_SIZE = os.environ.get("DRM_TIP_ADAPTER_BATCH_SIZE", "256")
WORKERS = os.environ.get("DRM_TIP_ADAPTER_WORKERS", "2")
WANDB_RUN_NAME = os.environ.get("DRM_TIP_ADAPTER_WANDB_RUN_NAME", "drm-wise-tip-adapter-full-data-vitb16-iwildcamval")


def assert_repo_supports_tip_adapter(repo_root):
    evaluator_path = Path(repo_root) / "src" / "eval_tail_cache.py"
    adapter_path = Path(repo_root) / "src" / "models" / "tip_adapter.py"
    if not evaluator_path.is_file() or not adapter_path.is_file():
        raise RuntimeError("The cloned repo lacks the Tip-Adapter control. Push the latest PoorFrogs code before rerunning.")
    evaluator_source = evaluator_path.read_text(encoding="utf-8")
    required_flags = ("--tip-adapter-beta-grid", "--tip-adapter-alpha-grid")
    if any(flag not in evaluator_source for flag in required_flags):
        raise RuntimeError("The cloned repo is stale and lacks Tip-Adapter runtime flags. Push the latest PoorFrogs code before rerunning.")


def build_command(data_location, checkpoint):
    return [
        sys.executable,
        "src/eval_tail_cache.py",
        "--model=ViT-B/16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_drm_template",
        f"--data-location={data_location}",
        f"--load={checkpoint}",
        f"--wise-eval-alpha={FIXED_WISE_ALPHA}",
        "--prototype-scale-grid=0",
        "--cache-tau-grid=0",
        "--tail-gamma-grid=0",
        "--gate-mode-grid=none",
        "--gate-strength-grid=0",
        "--sequence-consensus-grid=0",
        "--sctr-strength-grid=0",
        "--sctr-tail-protection-grid=0",
        "--multi-prototype-k-grid=1",
        "--max-cache-examples-per-class=0",
        f"--tip-adapter-beta-grid={TIP_ADAPTER_BETA_GRID}",
        f"--tip-adapter-alpha-grid={TIP_ADAPTER_ALPHA_GRID}",
        f"--tip-adapter-query-chunk-size={TIP_ADAPTER_QUERY_CHUNK_SIZE}",
        f"--tip-adapter-cache-chunk-size={TIP_ADAPTER_CACHE_CHUNK_SIZE}",
        "--summary-head=tip_adapter",
        "--audit-metadata",
        f"--batch-size={BATCH_SIZE}",
        f"--workers={WORKERS}",
        "--device=auto",
    ]


def main():
    ensure_deps()
    repo_root = clone_or_update(DEFAULT_GITHUB_REPO, DEFAULT_KAGGLE_WORKING_REPO)
    drm_repo = clone_or_update(DEFAULT_DRM_GITHUB_REPO, DEFAULT_DRM_REPO)
    configure_import_path(repo_root)
    ensure_local_package_installed(repo_root)
    patch_tail_cache_eval_guard(repo_root)
    patch_iwildcam_val()
    assert_repo_supports_tip_adapter(repo_root)

    data_location = prepare_iwildcam_layout(repo_root)
    drm_checkpoint = find_drm_checkpoint()
    state_dict_path = Path("/kaggle/working/drm_iwildcam_vit_b16_state_dict.pt")
    converted_checkpoint = Path("/kaggle/working/drm_iwildcam_vit_b16_poorfrogs_clip_encoder.pt")
    export_drm_state_dict(drm_repo, drm_checkpoint, state_dict_path)
    build_poorfrogs_checkpoint(repo_root, state_dict_path, converted_checkpoint)

    command = build_command(data_location, converted_checkpoint)
    if configure_wandb():
        command.extend(["--wandb", "--wandb-project=PoorFrogs", f"--wandb-run-name={WANDB_RUN_NAME}"])
    else:
        command.append("--no-wandb")
    print("Running DRM + WiSE + full-data Tip-Adapter control:")
    print(" ".join(str(part) for part in command))
    run(command, cwd=repo_root)


if __name__ == "__main__":
    main()
