from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


class StpConfirmationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ConfirmationReady:
    spec_sha256: str
    source_bundle_sha256: str
    audit_manifest_sha256: str
    class_mapping_sha256: str
    ledger_genesis_sha256: str
    source_files: tuple[str, ...]


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise StpConfirmationError(f"Source path must be workspace-relative: {value!r}.")
    return path


def build_source_bundle_checksum(root: Path, source_files: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for source_file in sorted(source_files):
        relative_path = _safe_relative_path(source_file)
        path = root / relative_path
        if not path.is_file():
            raise StpConfirmationError(f"Source bundle file is missing: {relative_path}.")
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    if not path.is_file():
        raise StpConfirmationError(f"Required checksum file is missing: {path}.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, object]:
    if not path.is_file():
        raise StpConfirmationError(f"{label} is missing: {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StpConfirmationError(f"{label} is not valid JSON: {path}.") from error
    if not isinstance(payload, dict):
        raise StpConfirmationError(f"{label} must contain a JSON object.")
    return payload


def _required_string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise StpConfirmationError(f"{label} must include non-empty {key!r}.")
    return value


def _source_files(payload: dict[str, object]) -> tuple[str, ...]:
    raw_files = payload.get("source_files")
    if not isinstance(raw_files, list) or not raw_files or not all(isinstance(item, str) for item in raw_files):
        raise StpConfirmationError("Frozen specification must include non-empty source_files.")
    return tuple(raw_files)


def _spec_checksum(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def verify_confirmation_ready(spec_path: Path, ledger_path: Path, workspace_root: Path) -> ConfirmationReady:
    spec = _load_json(spec_path, "Frozen specification")
    if spec.get("confirmation_status") != "frozen":
        raise StpConfirmationError("Frozen specification must set confirmation_status to 'frozen'.")
    source_files = _source_files(spec)
    declared_bundle = _required_string(spec, "source_bundle_sha256", "Frozen specification")
    actual_bundle = build_source_bundle_checksum(workspace_root, source_files)
    if actual_bundle != declared_bundle:
        raise StpConfirmationError("Frozen specification source bundle checksum does not match the workspace.")
    manifest_checksum = _required_string(spec, "audit_manifest_sha256", "Frozen specification")
    class_checksum = _required_string(spec, "class_mapping_sha256", "Frozen specification")
    manifest_path = _safe_relative_path(_required_string(spec, "audit_manifest_path", "Frozen specification"))
    class_mapping_path = _safe_relative_path(_required_string(spec, "class_mapping_path", "Frozen specification"))
    if file_sha256(workspace_root / manifest_path) != manifest_checksum:
        raise StpConfirmationError("Frozen specification audit manifest checksum does not match the workspace.")
    if file_sha256(workspace_root / class_mapping_path) != class_checksum:
        raise StpConfirmationError("Frozen specification class mapping checksum does not match the workspace.")
    ledger_genesis_checksum = _required_string(spec, "confirmation_ledger_genesis_sha256", "Frozen specification")
    ledger = _load_json(ledger_path, "Confirmation ledger")
    current_ledger_checksum = file_sha256(ledger_path)
    ledger_genesis = ledger.get("genesis_ledger_sha256")
    if current_ledger_checksum != ledger_genesis_checksum and ledger_genesis != ledger_genesis_checksum:
        raise StpConfirmationError("Confirmation ledger does not descend from the frozen genesis ledger.")
    confirmations = ledger.get("confirmations")
    if not isinstance(confirmations, list):
        raise StpConfirmationError("Confirmation ledger must contain a confirmations list.")
    spec_checksum = _spec_checksum(spec)
    for receipt in confirmations:
        if isinstance(receipt, dict) and receipt.get("spec_sha256") == spec_checksum:
            raise StpConfirmationError("Confirmation ledger already contains a receipt for this frozen specification.")
    return ConfirmationReady(
        spec_sha256=spec_checksum,
        source_bundle_sha256=actual_bundle,
        audit_manifest_sha256=manifest_checksum,
        class_mapping_sha256=class_checksum,
        ledger_genesis_sha256=ledger_genesis_checksum,
        source_files=source_files,
    )


def publish_confirmation_output(
    temporary_output: Path,
    final_output: Path,
    ready: ConfirmationReady,
    *,
    git_commit: str,
    output_sha256: str,
    wandb_run_id: str | None,
) -> Path:
    if not temporary_output.is_dir():
        raise StpConfirmationError("Temporary confirmation output directory is missing.")
    if final_output.exists():
        raise StpConfirmationError("Confirmation output already exists and cannot be replaced.")
    receipt = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": ready.spec_sha256,
        "source_bundle_sha256": ready.source_bundle_sha256,
        "audit_manifest_sha256": ready.audit_manifest_sha256,
        "class_mapping_sha256": ready.class_mapping_sha256,
        "confirmation_ledger_genesis_sha256": ready.ledger_genesis_sha256,
        "git_commit": git_commit,
        "output_sha256": output_sha256,
        "wandb_run_id": wandb_run_id,
    }
    receipt_path = temporary_output / "confirmation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    final_output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary_output, final_output)
    return final_output / receipt_path.name


def copy_receipt_to_next_ledger(receipt_path: Path, previous_ledger_path: Path, output_ledger_path: Path) -> None:
    ledger = _load_json(previous_ledger_path, "Confirmation ledger")
    receipt = _load_json(receipt_path, "Confirmation receipt")
    confirmations = ledger.get("confirmations")
    if not isinstance(confirmations, list):
        raise StpConfirmationError("Confirmation ledger must contain a confirmations list.")
    next_ledger = {"confirmations": [*confirmations, receipt]}
    next_ledger["genesis_ledger_sha256"] = ledger.get("genesis_ledger_sha256") or file_sha256(previous_ledger_path)
    output_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    output_ledger_path.write_text(json.dumps(next_ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_empty_ledger(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('{"confirmations": []}\n', encoding="utf-8")
