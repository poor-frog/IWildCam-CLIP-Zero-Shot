import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
SOURCE_COMMIT = "a97bfa5af010096701fe43a08e0f24678123353b"
TARGET_SEED = 20260725
VALIDATION_SEEDS = (20260721, 20260722, 20260723, 20260724, 20260725)
WORKING_REPOSITORY = Path("/kaggle/working/pgf-v0-source")
OUTPUT_ROOT = Path(f"/kaggle/working/paper-grade-training-foundation-v0-seed-{TARGET_SEED}")


def run(command):
    printable = [str(part) for part in command]
    print("+", " ".join(printable), flush=True)
    subprocess.check_call(printable)


def main():
    if WORKING_REPOSITORY.exists():
        raise FileExistsError(f"Refusing to reuse source checkout: {WORKING_REPOSITORY}")
    run(["git", "clone", "--no-checkout", GITHUB_REPOSITORY, WORKING_REPOSITORY])
    run(["git", "-C", WORKING_REPOSITORY, "checkout", "--detach", SOURCE_COMMIT])
    observed_commit = subprocess.check_output(
        ["git", "-C", str(WORKING_REPOSITORY), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if observed_commit != SOURCE_COMMIT:
        raise RuntimeError(f"Frozen source mismatch: {observed_commit} != {SOURCE_COMMIT}")
    sys.path.insert(0, str(WORKING_REPOSITORY))

    import src.pgf_pilot_artifacts as artifact_module
    import src.pgf_single_seed_kaggle as runner

    artifact_seeds = artifact_module.PILOT_SEEDS
    runner_seeds = runner.PILOT_SEEDS
    try:
        artifact_module.PILOT_SEEDS = VALIDATION_SEEDS
        runner.PILOT_SEEDS = VALIDATION_SEEDS
        runner.execute_single_seed(TARGET_SEED, WORKING_REPOSITORY, OUTPUT_ROOT, SOURCE_COMMIT)
    finally:
        artifact_module.PILOT_SEEDS = artifact_seeds
        runner.PILOT_SEEDS = runner_seeds


if __name__ == "__main__":
    main()
