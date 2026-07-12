import json
import importlib.util
from pathlib import Path

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
    assert '"--cd_path=prompts/iwildcam_cd.json"' in source
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
