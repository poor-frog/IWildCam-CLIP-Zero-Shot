import importlib.util
import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).parents[1]
SOURCE_COMMIT = "a97bfa5af010096701fe43a08e0f24678123353b"
PILOT_SEEDS = (20260721, 20260722, 20260723)
COMPLETION_SEEDS = (20260724, 20260725)
SEEDS = PILOT_SEEDS + COMPLETION_SEEDS
ALLOWED_OWNERS = {"klinh1912", "thanhquang71"}


def package_root(seed):
    return REPO_ROOT / f"kaggle-paper-grade-training-foundation-v0-seed-{seed}"


def load_launcher(seed):
    path = package_root(seed) / "kaggle_main.py"
    spec = importlib.util.spec_from_file_location(f"pgf_seed_{seed}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("seed", SEEDS)
def test_package_is_private_gpu_single_seed_kernel(seed):
    root = package_root(seed)
    metadata = json.loads((root / "kernel-metadata.json").read_text(encoding="utf-8"))
    launcher = load_launcher(seed)

    owner, slug = metadata["id"].split("/", 1)
    assert owner in ALLOWED_OWNERS
    assert slug == f"poorfrogs-pgf-v0-seed-{seed}"
    title_slug = re.sub(r"[^a-z0-9]+", "-", metadata["title"].lower()).strip("-")
    assert title_slug == slug
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["dataset_sources"] == ["thanhquang71/iwildcam-v2-0-2020-wilds-dataset"]
    assert launcher.SOURCE_COMMIT == SOURCE_COMMIT
    assert launcher.TARGET_SEED == seed
    assert str(seed) in str(launcher.OUTPUT_ROOT)
    if seed in COMPLETION_SEEDS:
        assert owner == "thanhquang71"
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert launcher.VALIDATION_SEEDS == SEEDS


@pytest.mark.parametrize("seeds", (PILOT_SEEDS, COMPLETION_SEEDS))
def test_packages_in_each_stage_differ_only_by_seed_specific_values(seeds):
    normalized_sources = []
    for seed in seeds:
        source = (package_root(seed) / "kaggle_main.py").read_text(encoding="utf-8")
        normalized_sources.append(re.sub(r"TARGET_SEED = \d+", "TARGET_SEED = <SEED>", source))

    assert len(set(normalized_sources)) == 1


@pytest.mark.parametrize("seed", SEEDS)
def test_launcher_dispatches_exactly_one_seed(monkeypatch, seed, tmp_path):
    launcher = load_launcher(seed)
    repository = tmp_path / "repo"
    calls = []

    monkeypatch.setattr(launcher, "WORKING_REPOSITORY", repository)
    monkeypatch.setattr(launcher, "OUTPUT_ROOT", tmp_path / f"output-{seed}")

    def run(command):
        calls.append(command)
        if command[:2] == ["git", "clone"]:
            repository.mkdir(parents=True)

    monkeypatch.setattr(launcher, "run", run)
    monkeypatch.setattr(
        launcher.subprocess,
        "check_output",
        lambda *args, **kwargs: SOURCE_COMMIT + "\n",
    )

    fake_module = type(
        "FakeModule",
        (),
        {
            "PILOT_SEEDS": PILOT_SEEDS,
            "execute_single_seed": staticmethod(
                lambda observed_seed, observed_repository, output_root, source_commit: calls.append(
                    ("execute", observed_seed, observed_repository, output_root, source_commit)
                )
            )
        },
    )
    monkeypatch.setitem(__import__("sys").modules, "src.pgf_single_seed_kaggle", fake_module)

    launcher.main()

    execution = calls[-1]
    assert execution[0] == "execute"
    assert execution[1] == seed
    assert execution[4] == SOURCE_COMMIT
