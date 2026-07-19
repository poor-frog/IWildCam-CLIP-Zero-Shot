from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from src.models.stp_confirmation import StpConfirmationError, build_source_bundle_checksum, file_sha256


class LesviConfirmationError(StpConfirmationError):
    pass


@dataclass(frozen=True, slots=True)
class LesviConfirmationReady:
    spec_sha256: str
    source_bundle_sha256: str
    audit_manifest_sha256: str
    viability_report_sha256: str
    class_mapping_sha256: str
    prior_artifact_sha256: str
    synthetic_verification_sha256: str
    checkpoint_sha256: str
    ledger_genesis_sha256: str
    source_files: tuple[str, ...]


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise LesviConfirmationError(f"{label} is missing: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LesviConfirmationError(f"{label} is not valid JSON: {path}.") from error
    if not isinstance(payload, dict):
        raise LesviConfirmationError(f"{label} must contain a JSON object.")
    return payload


def _required_string(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LesviConfirmationError(f"{label} must include non-empty {key!r}.")
    return value


def _spec_checksum(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_lesvi_confirmation_ready(
    *,
    spec_path: Path,
    ledger_path: Path,
    workspace_root: Path,
    audit_manifest_path: Path,
    viability_report_path: Path,
    class_mapping_path: Path,
    prior_artifact_path: Path,
    synthetic_verification_path: Path,
    checkpoint_path: Path,
    final_output_path: Path | None = None,
) -> LesviConfirmationReady:
    spec = _load_json(spec_path, "LESVI frozen specification")
    if spec.get("method") != "LESVI-v0" or spec.get("confirmation_status") != "frozen":
        raise LesviConfirmationError("LESVI specification must be frozen for method 'LESVI-v0'.")
    raw_sources = spec.get("source_files")
    if not isinstance(raw_sources, list) or not raw_sources or not all(isinstance(value, str) for value in raw_sources):
        raise LesviConfirmationError("LESVI specification must include source_files.")
    sources = tuple(raw_sources)
    actual_source = build_source_bundle_checksum(workspace_root, sources)
    if actual_source != _required_string(spec, "source_bundle_sha256", "LESVI frozen specification"):
        raise LesviConfirmationError("LESVI source bundle checksum does not match the workspace.")
    checks = (
        (audit_manifest_path, "audit_manifest_sha256", "audit manifest"),
        (viability_report_path, "viability_report_sha256", "confirmation viability report"),
        (class_mapping_path, "class_mapping_sha256", "class mapping"),
        (prior_artifact_path, "prior_artifact_sha256", "prior artifact"),
        (synthetic_verification_path, "synthetic_verification_sha256", "synthetic verification"),
        (checkpoint_path, "checkpoint_sha256", "checkpoint"),
    )
    actual: dict[str, str] = {}
    for path, key, label in checks:
        checksum = file_sha256(path)
        actual[key] = checksum
        if checksum != _required_string(spec, key, "LESVI frozen specification"):
            raise LesviConfirmationError(f"LESVI {label} checksum does not match the frozen specification.")
    ledger = _load_json(ledger_path, "LESVI confirmation ledger")
    genesis = _required_string(spec, "confirmation_ledger_genesis_sha256", "LESVI frozen specification")
    current_ledger_checksum = file_sha256(ledger_path)
    if current_ledger_checksum != genesis and ledger.get("genesis_ledger_sha256") != genesis:
        raise LesviConfirmationError("LESVI confirmation ledger does not descend from the frozen genesis ledger.")
    confirmations = ledger.get("confirmations")
    if not isinstance(confirmations, list):
        raise LesviConfirmationError("LESVI confirmation ledger must contain confirmations.")
    spec_sha = _spec_checksum(spec)
    if any(isinstance(receipt, dict) and receipt.get("spec_sha256") == spec_sha for receipt in confirmations):
        raise LesviConfirmationError("LESVI confirmation already has a successful receipt for this specification.")
    if final_output_path is not None and final_output_path.exists():
        raise LesviConfirmationError("LESVI final output already exists; confirmation cannot be repeated.")
    return LesviConfirmationReady(
        spec_sha256=spec_sha,
        source_bundle_sha256=actual_source,
        audit_manifest_sha256=actual["audit_manifest_sha256"],
        viability_report_sha256=actual["viability_report_sha256"],
        class_mapping_sha256=actual["class_mapping_sha256"],
        prior_artifact_sha256=actual["prior_artifact_sha256"],
        synthetic_verification_sha256=actual["synthetic_verification_sha256"],
        checkpoint_sha256=actual["checkpoint_sha256"],
        ledger_genesis_sha256=genesis,
        source_files=sources,
    )


def publish_lesvi_confirmation(
    *,
    temporary_output: Path,
    final_output: Path,
    ready: LesviConfirmationReady,
    previous_ledger_path: Path,
    next_ledger_path: Path,
    git_commit: str,
    output_sha256: str,
    metric_support_sha256: str,
    fallback_summary: Mapping[str, object],
    wandb_run_id: str,
) -> Path:
    if not temporary_output.is_dir() or final_output.exists():
        raise LesviConfirmationError("LESVI temporary output is missing or final output already exists.")
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "LESVI-v0",
        "spec_sha256": ready.spec_sha256,
        "source_bundle_sha256": ready.source_bundle_sha256,
        "audit_manifest_sha256": ready.audit_manifest_sha256,
        "viability_report_sha256": ready.viability_report_sha256,
        "class_mapping_sha256": ready.class_mapping_sha256,
        "prior_artifact_sha256": ready.prior_artifact_sha256,
        "synthetic_verification_sha256": ready.synthetic_verification_sha256,
        "checkpoint_sha256": ready.checkpoint_sha256,
        "confirmation_ledger_genesis_sha256": ready.ledger_genesis_sha256,
        "metric_support_sha256": metric_support_sha256,
        "git_commit": git_commit,
        "output_sha256": output_sha256,
        "fallback_summary": dict(fallback_summary),
        "wandb_run_id": wandb_run_id,
    }
    receipt_path = temporary_output / "confirmation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, final_output)

    ledger = _load_json(previous_ledger_path, "LESVI confirmation ledger")
    confirmations = ledger.get("confirmations")
    if not isinstance(confirmations, list):
        raise LesviConfirmationError("LESVI confirmation ledger must contain confirmations.")
    next_ledger = {
        "confirmations": [*confirmations, receipt],
        "genesis_ledger_sha256": ledger.get("genesis_ledger_sha256") or file_sha256(previous_ledger_path),
    }
    next_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    next_ledger_path.write_text(json.dumps(next_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return final_output / "confirmation_receipt.json"


def deterministic_output_checksum(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
