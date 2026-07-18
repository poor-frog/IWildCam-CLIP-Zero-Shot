import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-flyp-vfep-v0"


def load_launcher():
    spec = importlib.util.spec_from_file_location("kaggle_flyp_vfep", PACKAGE_ROOT / "kaggle_main.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_vfep_launcher_locks_phase_a_and_has_no_hardcoded_key():
    launcher = load_launcher()
    command = launcher.build_command("/kaggle/working/data", "/kaggle/input/checkpoint/flyp.pt", False)
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert "--vfep-pilot-output-dir=outputs/flyp-vfep-v0-val-audit" in command
    assert "--vfep-strength-grid=0,0.25,0.5,1" in command
    assert "--vfep-stp-strength-grid=0,0.25,0.5,1" in command
    assert "--vfep-shuffle-count=20" in command
    assert "--eval-datasets=IWildCamVal" in command
    assert "HARDCODED_WANDB_API_KEY" not in source
    assert "drm" not in source.lower()
    assert metadata["enable_gpu"] is True
