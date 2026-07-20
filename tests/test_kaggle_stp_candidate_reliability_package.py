import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "kaggle-stp-candidate-reliability-audit-v0"


def test_kernel_metadata_is_private_gpu_and_uses_frozen_inputs():
    metadata = json.loads((PACKAGE / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "klinh1912/poorfrogs-stp-candidate-reliability-audit-v0"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert "klinh1912/drm-iwildcam-vitb16-checkpoint" in metadata["dataset_sources"]


def test_launcher_locks_protocol_and_contains_no_credential():
    source = (PACKAGE / "kaggle_main.py").read_text(encoding="utf-8")

    assert '"DRM_WISE_EVAL_ALPHA": "0.2"' in source
    assert '"DRM_STP_CANDIDATE_RELIABILITY_BOOTSTRAP_SAMPLES": "2000"' in source
    assert '"DRM_STP_CANDIDATE_RELIABILITY_SHUFFLE_COUNT": "20"' in source
    assert '"DRM_STP_CANDIDATE_RELIABILITY_PERMUTATION_COUNT": "20"' in source
    assert "HARDCODED_WANDB_API_KEY" not in source


def test_generic_adapter_forces_candidate_audit_to_run_without_wandb():
    source = (ROOT / "kaggle_eval_drm_stmp_adapter.py").read_text(encoding="utf-8")
    guard = "STP_MECHANISM_AUDIT_OUTPUT_DIR or STP_ORACLE_AUDIT_OUTPUT_DIR or STP_CANDIDATE_RELIABILITY_OUTPUT_DIR"

    assert guard in source
    assert 'command.append("--no-wandb")' in source[source.rindex(f"if {guard}:"):]
