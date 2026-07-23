import json
from pathlib import Path

import pytest

from src.pgf_pilot_artifacts import (
    EXPECTED_AMP_POLICY,
    EXPECTED_PACKAGE_VERSIONS,
    PILOT_SEEDS,
    aggregate_pilot_outputs,
    build_pilot_manifest,
    build_single_seed_manifest,
    sha256_file,
    sha256_json,
    verify_seed_run,
)


SOURCE_COMMIT = "a" * 40


def _write_seed_output(root: Path, seed: int, suffix: str) -> Path:
    run_directory = root / f"seed-{seed}"
    run_directory.mkdir(parents=True)
    best = run_directory / "best_checkpoint.pt"
    final = run_directory / "final_checkpoint.pt"
    best.write_bytes(f"best-{suffix}".encode())
    final.write_bytes(f"final-{suffix}".encode())
    validation_trace = [{"epoch": epoch, "metric_value": epoch / 100} for epoch in range(1, 21)]
    wise_trace = [{"alpha": alpha, "metric_value": alpha} for alpha in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]
    runtime = {
        "seed": seed,
        "source_commit": SOURCE_COMMIT,
        "package_versions": EXPECTED_PACKAGE_VERSIONS,
        "amp_policy": EXPECTED_AMP_POLICY,
        "gpu_names": ["Tesla T4", "Tesla T4"],
    }
    receipt = {
        "status": "complete",
        "seed": seed,
        "protocol_configuration_sha256": "one-config",
        "runtime_determinism": runtime,
        "training": {
            "best_epoch": 10,
            "best_validation_score": 0.4,
            "selected_wise_alpha": 0.1,
            "selected_wise_score": 0.41,
            "validation_trace": validation_trace,
            "wise_selection_trace": wise_trace,
            "amp_skipped_step_count": 0,
            "non_finite_loss_or_gradient_observed": False,
        },
        "provenance": {
            "source": {
                "git_commit": SOURCE_COMMIT,
                "source_tree_sha256": "one-source",
            },
            "dataset": {
                "dataset_snapshot_sha256": "one-dataset",
            },
            "checkpoints": {
                "best_validation": {
                    "path": f"/kaggle/working/seed-{seed}/best_checkpoint.pt",
                    "sha256": sha256_file(best),
                },
                "final_wise": {
                    "path": f"/kaggle/working/seed-{seed}/final_checkpoint.pt",
                    "sha256": sha256_file(final),
                },
            },
        },
        "split_firewall": {
            "accessed_datasets": ["IWildCam", "IWildCamVal"],
        },
        "validity": {
            "source_commit_matches": True,
            "dataset_provenance_present": True,
            "split_firewall_passed": True,
        },
    }
    (run_directory / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return run_directory


def test_verify_seed_run_replays_hashes_from_downloaded_output(tmp_path):
    run_directory = _write_seed_output(tmp_path, PILOT_SEEDS[0], "one")

    summary = verify_seed_run(run_directory, PILOT_SEEDS[0], SOURCE_COMMIT)
    manifest = build_single_seed_manifest(summary)

    assert summary["seed"] == PILOT_SEEDS[0]
    assert summary["receipt_path"] == f"seed-{PILOT_SEEDS[0]}/run_receipt.json"
    assert manifest["status"] == "complete"
    assert manifest["final_splits_opened"] is False


def test_verify_seed_run_rejects_modified_checkpoint(tmp_path):
    run_directory = _write_seed_output(tmp_path, PILOT_SEEDS[0], "one")
    (run_directory / "final_checkpoint.pt").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        verify_seed_run(run_directory, PILOT_SEEDS[0], SOURCE_COMMIT)


def test_build_pilot_manifest_requires_shared_provenance_and_unique_checkpoints(tmp_path):
    summaries = [
        verify_seed_run(_write_seed_output(tmp_path / str(seed), seed, str(index)), seed, SOURCE_COMMIT)
        for index, seed in enumerate(PILOT_SEEDS)
    ]

    manifest = build_pilot_manifest(summaries, SOURCE_COMMIT)

    assert manifest["status"] == "complete"
    assert manifest["seeds"] == list(PILOT_SEEDS)
    assert manifest["final_splits_opened"] is False

    summaries[1]["dataset_snapshot_sha256"] = "different"
    with pytest.raises(RuntimeError, match="dataset_snapshot_sha256"):
        build_pilot_manifest(summaries, SOURCE_COMMIT)


def test_aggregate_pilot_outputs_writes_manifest_once(tmp_path):
    outputs = {
        seed: _write_seed_output(tmp_path / str(seed), seed, str(index))
        for index, seed in enumerate(PILOT_SEEDS)
    }
    output_path = tmp_path / "pilot_manifest.json"

    written = aggregate_pilot_outputs(outputs, SOURCE_COMMIT, output_path)

    manifest = json.loads(written.read_text(encoding="utf-8"))
    assert manifest["seeds"] == list(PILOT_SEEDS)
    assert len({item["receipt_sha256"] for item in manifest["seed_receipts"]}) == 3
    with pytest.raises(FileExistsError):
        aggregate_pilot_outputs(outputs, SOURCE_COMMIT, output_path)


def test_runtime_environment_hash_excludes_only_seed_and_source_commit(tmp_path):
    first = verify_seed_run(
        _write_seed_output(tmp_path / "first", PILOT_SEEDS[0], "first"),
        PILOT_SEEDS[0],
        SOURCE_COMMIT,
    )
    second_directory = _write_seed_output(tmp_path / "second", PILOT_SEEDS[1], "second")
    second_receipt_path = second_directory / "run_receipt.json"
    second_receipt = json.loads(second_receipt_path.read_text(encoding="utf-8"))
    second_receipt["runtime_determinism"]["gpu_names"] = ["different GPU"]
    second_receipt_path.write_text(json.dumps(second_receipt), encoding="utf-8")
    second = verify_seed_run(second_directory, PILOT_SEEDS[1], SOURCE_COMMIT)

    assert first["runtime_environment_sha256"] != second["runtime_environment_sha256"]
    assert first["validation_trace_sha256"] == sha256_json(
        [{"epoch": epoch, "metric_value": epoch / 100} for epoch in range(1, 21)]
    )
