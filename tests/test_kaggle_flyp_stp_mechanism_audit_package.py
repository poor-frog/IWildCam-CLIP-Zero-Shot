import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-flyp-stp-mechanism-audit"


def load_launcher():
    spec = importlib.util.spec_from_file_location("kaggle_flyp_stp_audit", PACKAGE_ROOT / "kaggle_main.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_clean_flyp_audit_kernel_isolated_from_drm_and_wise():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True
    assert "drm-iwildcam-vitb16-checkpoint" not in metadata["dataset_sources"]
    assert "klinh1912/flyp-official-vitb16-checkpoint" in metadata["dataset_sources"]
    assert "--wise-eval-alpha" not in source
    assert 'HARDCODED_WANDB_API_KEY = ""' in source


def test_clean_flyp_audit_command_locks_val_audit_protocol():
    launcher = load_launcher()
    command = launcher.build_command("/kaggle/working/data", "/kaggle/input/checkpoint/flyp.pt", use_wandb=False)

    assert "--eval-datasets=IWildCamVal" in command
    assert "--stp-mechanism-audit-foundation=flyp" in command
    assert "--stp-mechanism-audit-bootstrap-samples=2000" in command
    assert "--sequence-consensus-grid=0" in command
    assert "--no-wandb" in command
    assert not any("wise" in item.lower() or "drm" in item.lower() or "ood" in item.lower() for item in command)
