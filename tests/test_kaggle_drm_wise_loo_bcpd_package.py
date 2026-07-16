import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-drm-wise-loo-bcpd-pilot"


def test_loo_bcpd_kernel_uses_drm_checkpoint_and_gpu():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["code_file"] == "kaggle_main.py"
    assert metadata["enable_gpu"] is True
    assert "klinh1912/drm-iwildcam-vitb16-checkpoint" in metadata["dataset_sources"]


def test_loo_bcpd_kernel_locks_pilot_configuration_and_exposes_private_key_override():
    source = (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")

    assert '"DRM_WISE_EVAL_ALPHA": "0.2"' in source
    assert '"DRM_LOO_BCPD_STRENGTH_GRID": "0,0.25,0.5,1"' in source
    assert '"DRM_STMP_SEQUENCE_CONSENSUS_GRID": "0,0.5"' in source
    assert 'HARDCODED_WANDB_API_KEY = ""' in source
    assert 'environment["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY' in source
