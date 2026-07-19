import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
ANNOTATION_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz"
ANNOTATION_ROOT = Path("/kaggle/working/cct20-annotations")
OUTPUT_DIR = Path("/kaggle/working/cct20-lesvi-freeze")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def prepare_annotations():
    archive = ANNOTATION_ROOT / "eccv_18_annotations.tar.gz"
    extracted = ANNOTATION_ROOT / "eccv_18_annotation_files"
    if not (extracted / "train_annotations.json").is_file():
        ANNOTATION_ROOT.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ANNOTATION_URL, archive)
        with tarfile.open(archive, "r:gz") as bundle:
            root = ANNOTATION_ROOT.resolve()
            for member in bundle.getmembers():
                target = (ANNOTATION_ROOT / member.name).resolve()
                if root not in target.parents and target != root:
                    raise ValueError("CCT annotation archive contains an unsafe path.")
            bundle.extractall(ANNOTATION_ROOT)
    return extracted


def ensure_repository_support():
    required = (
        WORKING_REPOSITORY / "src" / "prepare_lesvi_cct20.py",
        WORKING_REPOSITORY / "src" / "eval_lesvi_cct20.py",
        WORKING_REPOSITORY / "src" / "models" / "lesvi_cct20.py",
    )
    if not all(path.is_file() for path in required):
        raise RuntimeError("The cloned repository does not contain the frozen CCT-20 LESVI protocol.")


def main():
    clone_repository()
    ensure_repository_support()
    annotations = prepare_annotations()
    command = [
        sys.executable,
        "src/prepare_lesvi_cct20.py",
        f"--train-annotations={annotations / 'train_annotations.json'}",
        f"--cis-validation-annotations={annotations / 'cis_val_annotations.json'}",
        f"--trans-validation-annotations={annotations / 'trans_val_annotations.json'}",
        f"--trans-test-annotations={annotations / 'trans_test_annotations.json'}",
        "--image-root=/kaggle/input/cct20-images",
        f"--output-dir={OUTPUT_DIR}",
        f"--workspace-root={WORKING_REPOSITORY}",
    ]
    environment = dict(__import__("os").environ, PYTHONPATH=str(WORKING_REPOSITORY))
    run(command, cwd=WORKING_REPOSITORY, env=environment)
    print(f"Frozen CCT-20 LESVI artifacts: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
