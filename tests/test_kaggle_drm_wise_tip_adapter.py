import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-wise-tip-adapter-control"


def test_drm_tip_adapter_command_locks_wise_and_enables_only_tip_control():
    from kaggle_eval_drm_wise_tip_adapter import build_command

    command = build_command("/kaggle/working/data", "/kaggle/working/drm.pt")

    assert "--wise-eval-alpha=0.2" in command
    assert "--prototype-scale-grid=0" in command
    assert "--sequence-consensus-grid=0" in command
    assert "--tip-adapter-beta-grid=0.1,0.5,1,2,5,7" in command
    assert "--tip-adapter-alpha-grid=0.1,0.5,1,2,3" in command
    assert "--summary-head=tip_adapter" in command
    assert all("--wise-alpha-grid" not in argument for argument in command)


def test_tip_adapter_kernel_metadata_targets_dedicated_private_gpu_kernel():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True
    assert metadata["is_private"] is True
    assert "thanhquang71/iwildcam-v2-0-2020-wilds-dataset" in metadata["dataset_sources"]
    assert "klinh1912/drm-iwildcam-vitb16-checkpoint" in metadata["dataset_sources"]


def test_tip_adapter_kernel_wrapper_uses_secret_or_empty_hardcode_placeholder():
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert 'HARDCODED_WANDB_API_KEY = ""' in source
    assert '"kaggle_eval_drm_wise_tip_adapter.py"' in source
