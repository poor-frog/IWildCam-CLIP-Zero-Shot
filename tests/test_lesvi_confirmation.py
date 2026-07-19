import json
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.models.lesvi_confirmation import (
    LesviConfirmationError,
    deterministic_output_checksum,
    publish_lesvi_confirmation,
    verify_lesvi_confirmation_ready,
)
from src.models.stp_confirmation import build_source_bundle_checksum, file_sha256
from src.eval_lesvi_internal_confirm import _open_wandb_registry


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _guard_fixture(tmp_path: Path):
    _write(tmp_path / "source.py", "VALUE = 1\n")
    for name in ("audit.json", "viability.json", "mapping.json", "prior.json", "synthetic.json", "checkpoint.pt"):
        _write(tmp_path / name, "{}\n")
    _write(tmp_path / "ledger.json", '{"confirmations": []}\n')
    spec = {
        "method": "LESVI-v0",
        "confirmation_status": "frozen",
        "source_files": ["source.py"],
        "source_bundle_sha256": build_source_bundle_checksum(tmp_path, ["source.py"]),
        "audit_manifest_sha256": file_sha256(tmp_path / "audit.json"),
        "viability_report_sha256": file_sha256(tmp_path / "viability.json"),
        "class_mapping_sha256": file_sha256(tmp_path / "mapping.json"),
        "prior_artifact_sha256": file_sha256(tmp_path / "prior.json"),
        "synthetic_verification_sha256": file_sha256(tmp_path / "synthetic.json"),
        "checkpoint_sha256": file_sha256(tmp_path / "checkpoint.pt"),
        "confirmation_ledger_genesis_sha256": file_sha256(tmp_path / "ledger.json"),
    }
    _write(tmp_path / "spec.json", json.dumps(spec))
    return spec


def _verify(tmp_path: Path):
    return verify_lesvi_confirmation_ready(
        spec_path=tmp_path / "spec.json",
        ledger_path=tmp_path / "ledger.json",
        workspace_root=tmp_path,
        audit_manifest_path=tmp_path / "audit.json",
        viability_report_path=tmp_path / "viability.json",
        class_mapping_path=tmp_path / "mapping.json",
        prior_artifact_path=tmp_path / "prior.json",
        synthetic_verification_path=tmp_path / "synthetic.json",
        checkpoint_path=tmp_path / "checkpoint.pt",
        final_output_path=tmp_path / "final",
    )


def test_guard_rejects_source_change_before_confirmation(tmp_path):
    _guard_fixture(tmp_path)
    _write(tmp_path / "source.py", "VALUE = 2\n")
    with pytest.raises(LesviConfirmationError, match="source bundle"):
        _verify(tmp_path)


def test_receipt_ledger_refuses_second_success(tmp_path):
    _guard_fixture(tmp_path)
    ready = _verify(tmp_path)
    temporary = tmp_path / "temporary"
    temporary.mkdir()
    _write(temporary / "report.json", "{}\n")
    checksum = deterministic_output_checksum([temporary / "report.json"])
    receipt = publish_lesvi_confirmation(
        temporary_output=temporary,
        final_output=tmp_path / "published",
        ready=ready,
        previous_ledger_path=tmp_path / "ledger.json",
        next_ledger_path=tmp_path / "next-ledger.json",
        git_commit="abc",
        output_sha256=checksum,
        metric_support_sha256="support",
        fallback_summary={},
        wandb_run_id="lesvi-test",
    )
    assert receipt.is_file()
    with pytest.raises(LesviConfirmationError, match="successful receipt"):
        verify_lesvi_confirmation_ready(
            spec_path=tmp_path / "spec.json",
            ledger_path=tmp_path / "next-ledger.json",
            workspace_root=tmp_path,
            audit_manifest_path=tmp_path / "audit.json",
            viability_report_path=tmp_path / "viability.json",
            class_mapping_path=tmp_path / "mapping.json",
            prior_artifact_path=tmp_path / "prior.json",
            synthetic_verification_path=tmp_path / "synthetic.json",
            checkpoint_path=tmp_path / "checkpoint.pt",
            final_output_path=tmp_path / "another-final",
        )


def test_confirmation_runner_has_guard_before_heavy_imports():
    source = Path("src/eval_lesvi_internal_confirm.py").read_text(encoding="utf-8")
    guard_position = source.index("ready = verify_lesvi_confirmation_ready")
    registry_position = source.index("wandb_run = _open_wandb_registry")
    run_position = source.index("run_confirmation(args, ready, wandb_run)")
    assert guard_position < registry_position < run_position
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "wilds" not in top_level_imports
    assert "wandb" not in top_level_imports


def test_new_kaggle_launchers_do_not_embed_secrets():
    for path in (Path("kaggle-flyp-lesvi-freeze/kaggle_main.py"), Path("kaggle-flyp-lesvi-confirm/kaggle_main.py")):
        source = path.read_text(encoding="utf-8")
        assert "HARDCODED_WANDB" not in source
        assert "api_key=" not in source.lower()


def test_confirm_launcher_requires_explicit_latest_ledger():
    source = Path("kaggle-flyp-lesvi-confirm/kaggle_main.py").read_text(encoding="utf-8")
    assert 'require_configured_file("LESVI_CONFIRMATION_LEDGER")' in source
    assert 'find_file("LESVI_CONFIRMATION_LEDGER", "confirmation_genesis_ledger.json")' not in source


def test_wandb_registry_rejects_successful_receipt_before_confirmation(monkeypatch):
    class _Run:
        id = "lesvi-existing"
        summary = {"confirm/receipt_published": True}

        def finish(self, **_):
            return None

    captured = {}

    def init(**kwargs):
        captured.update(kwargs)
        return _Run()

    monkeypatch.setenv("WANDB_API_KEY", "test-only")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
    args = SimpleNamespace(wandb_project="PoorFrogs", wandb_run_name="lesvi")
    with pytest.raises(RuntimeError, match="durable registry"):
        _open_wandb_registry(args, "a" * 64)
    assert captured["id"] == "lesvi-" + "a" * 24
    assert captured["resume"] == "allow"
