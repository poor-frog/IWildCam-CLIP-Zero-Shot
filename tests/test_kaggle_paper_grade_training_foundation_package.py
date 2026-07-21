import importlib.util
import json
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).parents[1] / "kaggle-paper-grade-training-foundation-v0"
SOURCE_COMMIT = "cbf204d36eb4f19df6867705fd784785ca8b1be0"


def load_launcher():
    spec = importlib.util.spec_from_file_location("kaggle_pgf_v0", PACKAGE_ROOT / "kaggle_main.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    return launcher


def test_metadata_is_private_gpu_pilot_with_only_iwildcam_attached():
    metadata = json.loads((PACKAGE_ROOT / "kernel-metadata.json").read_text(encoding="utf-8"))

    assert metadata["id"] == "klinh1912/poorfrogs-paper-grade-training-foundation-v0"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == ["thanhquang71/iwildcam-v2-0-2020-wilds-dataset"]
    assert metadata["kernel_sources"] == []
    assert metadata["model_sources"] == []


def test_launcher_pins_validated_source_commit_and_three_preregistered_seeds():
    launcher = load_launcher()

    assert launcher.SOURCE_COMMIT == SOURCE_COMMIT
    assert launcher.PILOT_SEEDS == (20260721, 20260722, 20260723)
    assert "pull" not in (PACKAGE_ROOT / "kaggle_main.py").read_text(encoding="utf-8")


def test_launcher_pins_every_installed_dependency():
    launcher = load_launcher()

    assert launcher.PINNED_DEPENDENCIES
    assert all("==" in dependency for dependency in launcher.PINNED_DEPENDENCIES)
    assert launcher.EXPECTED_PACKAGE_VERSIONS["open-clip-torch"] == "3.3.0"
    assert launcher.EXPECTED_PACKAGE_VERSIONS["wilds"] == "2.0.0"


def test_commands_lock_frozen_training_configuration_and_unique_paths():
    launcher = load_launcher()
    commands = [launcher.build_command(seed, "/kaggle/working/data", False) for seed in launcher.PILOT_SEEDS]
    required = {
        "--paper-grade-training-foundation-v0",
        "--deterministic-training",
        "--model=ViT-B-16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--template=iwildcam_template",
        "--epochs=20",
        "--batch-size=256",
        "--workers=4",
        "--lr=0.00001",
        "--wd=0.2",
        "--lr-scheduler=cosine",
        "--warmup-length=0",
        "--maple-precision=amp",
        "--best-metric=F1-macro_all",
        "--drm-weight=0",
        "--tail-proto-weight=0",
        "--btel-weight=0",
        "--wise-alphas=0,0.05,0.1,0.15,0.2,0.3",
        "--no-wandb",
    }
    artifact_arguments = []
    for seed, command in zip(launcher.PILOT_SEEDS, commands):
        assert required <= set(command)
        assert f"--seed={seed}" in command
        assert not any(argument.startswith("--load=") for argument in command)
        assert not any(argument.startswith("--eval-datasets=") for argument in command)
        seed_artifacts = [
            argument
            for argument in command
            if argument.startswith(("--determinism-receipt=", "--save=", "--best-checkpoint="))
        ]
        assert len(seed_artifacts) == 3
        assert all(str(seed) in argument for argument in seed_artifacts)
        artifact_arguments.extend(seed_artifacts)
    assert len(set(artifact_arguments)) == 9


def _seed_receipt(seed, suffix):
    return {
        "seed": seed,
        "receipt_path": f"/working/seed-{seed}/run_receipt.json",
        "receipt_sha256": f"receipt-{suffix}",
        "source_tree_sha256": "one-source",
        "dataset_snapshot_sha256": "one-dataset",
        "protocol_configuration_sha256": "one-config",
        "runtime_environment_sha256": "one-environment",
        "best_checkpoint_sha256": f"best-{suffix}",
        "final_checkpoint_sha256": f"final-{suffix}",
        "best_epoch": 10,
        "best_validation_score": 0.4,
        "selected_wise_alpha": 0.1,
        "selected_wise_score": 0.41,
        "amp_skipped_step_count": 0,
        "non_finite_loss_or_gradient_observed": False,
        "validation_trace_sha256": f"validation-{suffix}",
        "wise_selection_trace_sha256": f"wise-{suffix}",
    }


def test_pilot_manifest_requires_shared_provenance_and_unique_checkpoints():
    launcher = load_launcher()
    receipts = [_seed_receipt(seed, index) for index, seed in enumerate(launcher.PILOT_SEEDS)]

    manifest = launcher.build_pilot_manifest(receipts)

    assert manifest["status"] == "complete"
    assert manifest["source_commit"] == SOURCE_COMMIT
    assert manifest["selection_split"] == "IWildCamVal"
    assert manifest["final_splits_opened"] is False

    receipts[1]["dataset_snapshot_sha256"] = "different-dataset"
    with pytest.raises(RuntimeError, match="dataset_snapshot_sha256"):
        launcher.build_pilot_manifest(receipts)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("best_epoch", None, "best epoch"),
        ("selected_wise_alpha", None, "WiSE alpha"),
        ("amp_skipped_step_count", 1, "skipped AMP"),
        ("non_finite_loss_or_gradient_observed", True, "non-finite"),
    ],
)
def test_pilot_manifest_rejects_failed_training_gate(field, value, message):
    launcher = load_launcher()
    receipts = [_seed_receipt(seed, index) for index, seed in enumerate(launcher.PILOT_SEEDS)]
    receipts[0][field] = value

    with pytest.raises(RuntimeError, match=message):
        launcher.build_pilot_manifest(receipts)


def test_launcher_main_dispatches_exactly_three_val_only_runs(monkeypatch, tmp_path):
    launcher = load_launcher()
    output_root = tmp_path / "pilot"
    commands = []
    verified = []
    manifest_inputs = []

    monkeypatch.setattr(launcher, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(launcher, "clone_frozen_repository", lambda: None)
    monkeypatch.setattr(launcher, "ensure_frozen_source_support", lambda: None)
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda: None)
    monkeypatch.setattr(launcher, "prepare_iwildcam_layout", lambda: tmp_path / "data")
    monkeypatch.setattr(launcher, "configure_wandb", lambda: False)

    def run(command, cwd=None, env=None):
        commands.append(command)

    def verify_seed_receipt(seed):
        verified.append(seed)
        return _seed_receipt(seed, len(verified))

    def write_pilot_manifest(receipts):
        manifest_inputs.extend(receipts)

    monkeypatch.setattr(launcher, "run", run)
    monkeypatch.setattr(launcher, "verify_seed_receipt", verify_seed_receipt)
    monkeypatch.setattr(launcher, "write_pilot_manifest", write_pilot_manifest)

    launcher.main()

    assert len(commands) == 3
    assert verified == list(launcher.PILOT_SEEDS)
    assert [item["seed"] for item in manifest_inputs] == list(launcher.PILOT_SEEDS)
    assert all("--val-dataset=IWildCamVal" in command for command in commands)
    assert all(not any(argument.startswith("--eval-datasets=") for argument in command) for command in commands)
