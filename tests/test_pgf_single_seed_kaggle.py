from pathlib import Path

import pytest

from src.pgf_pilot_artifacts import PILOT_SEEDS
from src.pgf_single_seed_kaggle import build_command, execute_single_seed


def test_build_command_locks_one_preregistered_seed_and_val_only_paths(tmp_path):
    seed = PILOT_SEEDS[0]
    command = build_command(seed, tmp_path / "repo", tmp_path / "output", tmp_path / "data")

    assert f"--seed={seed}" in command
    assert "--epochs=20" in command
    assert "--maple-precision=amp" in command
    assert "--val-dataset=IWildCamVal" in command
    assert "--wise-alphas=0,0.05,0.1,0.15,0.2,0.3" in command
    assert "--no-wandb" in command
    assert not any(str(argument).startswith("--eval-datasets=") for argument in command)
    assert all(
        str(seed) in str(argument)
        for argument in command
        if str(argument).startswith(("--determinism-receipt=", "--save=", "--best-checkpoint="))
    )


def test_build_command_rejects_non_preregistered_seed(tmp_path):
    with pytest.raises(ValueError, match="preregistered pilot set"):
        build_command(123, tmp_path / "repo", tmp_path / "output", tmp_path / "data")


def test_execute_single_seed_runs_then_verifies_and_writes_manifest(monkeypatch, tmp_path):
    import src.pgf_single_seed_kaggle as launcher

    repository = tmp_path / "repo"
    train_source = repository / "src" / "train_flyp.py"
    train_source.parent.mkdir(parents=True)
    train_source.write_text("PAPER_GRADE_AMP_INIT_SCALE = 1.0\n", encoding="utf-8")
    output_root = tmp_path / "output"
    seed = PILOT_SEEDS[0]
    calls = []
    monkeypatch.setattr(launcher, "ensure_dependencies", lambda repository: calls.append("dependencies"))
    monkeypatch.setattr(launcher, "prepare_iwildcam_layout", lambda repository: tmp_path / "data")

    def run(command, cwd=None, env=None):
        calls.append(("run", command, cwd, env))

    monkeypatch.setattr(launcher, "run", run)
    monkeypatch.setattr(
        launcher,
        "verify_seed_run",
        lambda run_directory, observed_seed, source_commit: {
            "seed": observed_seed,
            "source_commit": source_commit,
        },
    )
    monkeypatch.setattr(
        launcher,
        "build_single_seed_manifest",
        lambda summary: {"status": "complete", "seed": summary["seed"]},
    )

    manifest = execute_single_seed(seed, repository, output_root, "a" * 40)

    assert calls[0] == "dependencies"
    assert calls[1][0] == "run"
    assert Path(manifest).is_file()
    assert str(seed) in " ".join(str(part) for part in calls[1][1])
    with pytest.raises(FileExistsError):
        execute_single_seed(seed, repository, output_root, "a" * 40)
