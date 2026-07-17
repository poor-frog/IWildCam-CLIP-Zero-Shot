import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-flyp-selective-stp-v0"


def load_launcher():
    spec = importlib.util.spec_from_file_location("kaggle_flyp_selective_stp", PACKAGE_ROOT / "kaggle_main.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_selective_stp_kernel_is_clean_flyp_only():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True
    assert "drm-iwildcam-vitb16-checkpoint" not in metadata["dataset_sources"]
    assert "--wise-eval-alpha" not in source
    assert "kaggle_eval_drm_stmp_adapter.py" not in source
    assert 'HARDCODED_WANDB_API_KEY = ""' in source


def test_selective_stp_command_locks_the_frozen_rule():
    launcher = load_launcher()
    command = launcher.build_command("/kaggle/working/data", "/kaggle/input/clean/flyp.pt", use_wandb=False)

    assert "--eval-datasets=IWildCamVal" in command
    assert "--prototype-scale-grid=50" in command
    assert "--sequence-consensus-grid=0,0.5" in command
    assert "--stp-selective-target" in command
    assert "--stp-selective-eta=0.5" in command
    assert "--stp-selective-margin-threshold=0.9341613054275513" in command
    assert "--stp-selective-min-burst-length=3" in command
    assert "--selection-output=/kaggle/working/selective_stp_selection.json" in command
    assert "--no-wandb" in command
    assert not any("wise" in item.lower() or "drm" in item.lower() for item in command)
