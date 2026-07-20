import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments/drm_wise_stp_v0/reproducibility_manifest.json"


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def git_file_at_commit(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)


def test_manifest_locks_result_and_claim_boundary():
    manifest = load_manifest()

    assert manifest["result_run"]["run_id"] == "03gg65vx"
    assert manifest["source_checkout"]["commit"] == "c1d2290035285c512b2bcfd4e3a5c55f76fdbe1e"
    assert manifest["selected_configuration"]["wise_alpha"] == 0.2
    assert manifest["selected_configuration"]["sequence_eta"] == 0.5
    assert manifest["final_metrics"]["IWildCamOOD"]["macro_f1"] == 0.42423877120018005
    assert "multi-seed DRM result" in manifest["claim_boundary"]["disallowed"]
    assert len(manifest["wise_selection"]) == 10
    assert max(manifest["wise_selection"], key=lambda row: row["val_macro_f1"])["alpha"] == 0.2


def test_available_local_artifacts_match_manifest_hashes():
    manifest = load_manifest()
    source = manifest["checkpoint"]["source"]
    source_path = ROOT / source["local_path"]

    assert source_path.stat().st_size == source["size_bytes"]
    assert sha256_file(source_path) == source["sha256"]
    for artifact in manifest["local_evidence"]:
        artifact_path = ROOT / artifact["path"]
        assert artifact_path.stat().st_size == artifact["size_bytes"]
        assert sha256_file(artifact_path) == artifact["sha256"]


def test_source_files_match_recorded_result_commit():
    manifest = load_manifest()
    commit = manifest["source_checkout"]["commit"]

    for source_file in manifest["source_files"]:
        content = git_file_at_commit(commit, source_file["path"])
        assert sha256_bytes(content) == source_file["sha256_at_source_commit"]


def test_runtime_evaluator_patch_is_replayable():
    manifest = load_manifest()
    commit = manifest["source_checkout"]["commit"]
    evaluator = git_file_at_commit(commit, "src/eval_tail_cache.py").decode("utf-8")
    evaluator = evaluator.replace(
        "from src.train_flyp import clone_state_dict, ensure_open_clip_for_flyp",
        "from src.train_flyp import clone_state_dict",
    )
    evaluator = evaluator.replace("    ensure_open_clip_for_flyp(args.model)\n\n", "")
    evaluator = evaluator.replace(
        "--load must point to a FLYP CLIPEncoder checkpoint.",
        "--load must point to a CLIPEncoder checkpoint.",
    )

    assert sha256_bytes(evaluator.encode("utf-8")) == manifest["runtime_mutations"]["patched_evaluator_sha256"]
