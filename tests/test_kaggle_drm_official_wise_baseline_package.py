import json
import importlib.util
from pathlib import Path

import pytest
import torch


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-official-wise-baseline"


def load_launcher():
    path = PACKAGE_ROOT / "kaggle_main.py"
    spec = importlib.util.spec_from_file_location("official_wise_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_wise_package_uses_id_validation_and_dual_inference_source():
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert '"--train-dataset=IWildCamIDVal"' in source
    assert "concept_description_path()" in source
    assert '"--beta={BETA}"' in source
    assert 'from src.models.DRM_eval import drm_eval' in source
    assert '"IWildCamIDVal,IWildCamID,IWildCamOOD"' in source
    assert "--wise-alpha" not in source


def test_official_wise_metadata_targets_a_dedicated_kernel():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "huyphmhakq/poorfrogs-drm-official-wise-iwildcam"
    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True


def test_wise_interpolation_uses_zero_shot_weight_with_rho():
    launcher = load_launcher()
    zero_shot = torch.nn.Linear(1, 1, bias=False)
    finetuned = torch.nn.Linear(1, 1, bias=False)
    zero_shot.weight.data.fill_(2.0)
    finetuned.weight.data.fill_(10.0)

    merged = launcher.interpolate_wise(zero_shot.state_dict(), finetuned, 0.25)

    assert merged.weight.item() == 8.0


def test_wise_interpolation_rejects_mismatched_tensor_shapes():
    launcher = load_launcher()

    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, 2))

    with pytest.raises(RuntimeError, match="tensor shapes differ"):
        launcher.interpolate_wise({"weight": torch.zeros(1, 1)}, Dummy(), 0.25)


def test_drm_path_helpers_use_repo_root_and_restore_cwd(tmp_path, monkeypatch):
    launcher = load_launcher()
    prompt_path = tmp_path / "prompts" / "iwildcam_cd.json"
    prompt_path.parent.mkdir()
    prompt_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(launcher, "DRM_ROOT", tmp_path)

    assert launcher.concept_description_path() == prompt_path
    original = Path.cwd()
    with launcher.in_drm_repo():
        assert Path.cwd() == tmp_path
    assert Path.cwd() == original
