import importlib.util
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-wise-no-tpa-control"


def load_launcher():
    path = PACKAGE_ROOT / "kaggle_main.py"
    spec = importlib.util.spec_from_file_location("drm_wise_no_tpa_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_tpa_launcher_disables_prototype_residual_and_sequence_consensus():
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert 'env["DRM_STMP_PROTOTYPE_SCALE_GRID"] = "0"' in source
    assert 'env["DRM_STMP_SEQUENCE_CONSENSUS_GRID"] = "0"' in source
    assert 'env["DRM_STP_MULTI_PROTOTYPE_K_GRID"] = "1"' in source
    assert 'env["DRM_WISE_ALPHA_GRID"] = WISE_ALPHA_GRID' in source
    assert "HARDCODED_WANDB_API_KEY = \"\"" in source


def test_no_tpa_metadata_is_gpu_kernel_with_expected_datasets():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "huyphmhakq/poorfrogs-drm-wise-no-tpa-control-iwildcam"
    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == [
        "thanhquang71/iwildcam-v2-0-2020-wilds-dataset",
        "klinh1912/drm-iwildcam-vitb16-checkpoint",
    ]


def test_no_tpa_launcher_imports():
    launcher = load_launcher()

    assert launcher.WISE_ALPHA_GRID == "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
