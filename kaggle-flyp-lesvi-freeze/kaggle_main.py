import os
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
OUTPUT_DIR = Path("/kaggle/working/lesvi-v0-freeze")
CHECKPOINT_NAME = "flyp_official_b16_bs256_wd0p2_lr1e5_idval_best.pt"
BLOCKED_OUTPUT_DIR = Path("/kaggle/working/lesvi-v0-freeze-blocked")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def find_file(environment_name, filename):
    configured = os.environ.get(environment_name)
    if configured and Path(configured).is_file():
        return Path(configured)
    matches = sorted(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Attach {filename} or set {environment_name}.")
    return matches[0]


def find_data_location():
    configured = os.environ.get("LESVI_DATA_LOCATION")
    if configured and Path(configured).exists():
        return Path(configured)
    matches = sorted(Path("/kaggle/input").rglob("metadata.csv"))
    for metadata in matches:
        if "iwildcam" in str(metadata).lower():
            return metadata.parent.parent if metadata.parent.name == "iwildcam_v2.0" else metadata.parent
    raise FileNotFoundError("Attach IWildCam v2.0 or set LESVI_DATA_LOCATION.")


def publish_blocked_receipt(audit_path, viability_path):
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    viability = json.loads(Path(viability_path).read_text(encoding="utf-8"))
    if viability.get("viability_pass") is True:
        return False
    BLOCKED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    receipt = {
        "method": "LESVI-v0",
        "status": "blocked",
        "reason": "frozen_split_failed_viability_preflight",
        "confirmation_performance_materialized": False,
        "frozen_spec_created": False,
        "audit_manifest_sha256": hashlib.sha256(Path(audit_path).read_bytes()).hexdigest(),
        "viability_report_sha256": hashlib.sha256(Path(viability_path).read_bytes()).hexdigest(),
        "audit_phase": audit.get("phase"),
        "viability": viability,
    }
    output = BLOCKED_OUTPUT_DIR / "freeze_blocked_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"LESVI freeze blocked by preregistered viability gate; receipt written to {output}", flush=True)
    return True


def main():
    audit_manifest = find_file("LESVI_AUDIT_MANIFEST", "audit_manifest.json")
    viability_report = find_file("LESVI_VIABILITY_REPORT", "confirm_viability_aggregate.json")
    if publish_blocked_receipt(audit_manifest, viability_report):
        return
    clone_repository()
    run([sys.executable, "-m", "pip", "install", "-q", "wilds"])
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    environment = dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY))
    run([
        sys.executable,
        "src/prepare_lesvi_freeze.py",
        f"--data-location={find_data_location()}",
        f"--audit-manifest={audit_manifest}",
        f"--viability-report={viability_report}",
        f"--class-mapping={find_file('LESVI_CLASS_MAPPING', 'class_mapping.json')}",
        f"--confirmation-genesis-ledger={find_file('LESVI_GENESIS_LEDGER', 'confirmation_genesis_ledger.json')}",
        f"--checkpoint={find_file('LESVI_CHECKPOINT', CHECKPOINT_NAME)}",
        f"--output-dir={OUTPUT_DIR}",
        f"--workspace-root={WORKING_REPOSITORY}",
    ], cwd=WORKING_REPOSITORY, env=environment)


if __name__ == "__main__":
    main()
