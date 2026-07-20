from pathlib import Path
from types import SimpleNamespace

import pytest

from src.eval_tail_cache import _validate_stp_mechanism_audit_args


def audit_args(**overrides):
    values = {
        "stp_mechanism_audit_output_dir": Path("outputs/test-stp-audit-foundation"),
        "stp_mechanism_audit_foundation": "flyp",
        "val_dataset": "IWildCamVal",
        "eval_datasets": ["IWildCamVal"],
        "max_eval_batches": None,
        "max_train_batches": None,
        "stp_mechanism_audit_bootstrap_samples": 1,
        "stp_oracle_audit_output_dir": None,
        "stp_oracle_audit_bootstrap_samples": 1,
        "stp_oracle_audit_shuffle_count": 1,
        "wise_eval_alpha": None,
        "prototype_scale_grid": "50",
        "multi_prototype_k_grid": "1",
        "cd_path": None,
        "cache_tau_grid": "0",
        "tail_gamma_grid": "0",
        "gate_mode_grid": "none",
        "gate_strength_grid": "0",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_clean_flyp_audit_accepts_no_wise_interpolation():
    _validate_stp_mechanism_audit_args(audit_args())


def test_clean_flyp_audit_rejects_wise_interpolation():
    with pytest.raises(ValueError, match="must not apply WiSE"):
        _validate_stp_mechanism_audit_args(audit_args(wise_eval_alpha=0.2))


def test_oracle_audit_accepts_only_locked_drm_wise_val_audit_protocol():
    args = audit_args(
        stp_mechanism_audit_output_dir=None,
        stp_oracle_audit_output_dir=Path("outputs/test-stp-oracle-audit"),
        stp_mechanism_audit_foundation="drm_wise",
        wise_eval_alpha=0.2,
    )

    _validate_stp_mechanism_audit_args(args)

    with pytest.raises(ValueError, match="IWildCamVal"):
        _validate_stp_mechanism_audit_args(SimpleNamespace(**{**vars(args), "eval_datasets": ["IWildCamOOD"]}))
