import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "kaggle-stp-oracle-audit-v0"


def test_oracle_audit_kernel_is_private_gpu_and_uses_expected_inputs():
    metadata = json.loads((PACKAGE / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"].endswith("/poorfrogs-stp-oracle-audit-v0")
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert "klinh1912/drm-iwildcam-vitb16-checkpoint" in metadata["dataset_sources"]


def test_oracle_launcher_has_frozen_protocol_and_no_hardcoded_credential():
    source = (PACKAGE / "kaggle_main.py").read_text(encoding="utf-8")

    assert '"DRM_WISE_EVAL_ALPHA": "0.2"' in source
    assert '"DRM_STP_ORACLE_AUDIT_BOOTSTRAP_SAMPLES": "2000"' in source
    assert '"DRM_STP_ORACLE_AUDIT_SHUFFLE_COUNT": "20"' in source
    assert "HARDCODED_WANDB_API_KEY" not in source
