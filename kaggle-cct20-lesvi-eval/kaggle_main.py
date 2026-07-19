import json
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
ANNOTATION_URL = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/eccv_18_annotations.tar.gz"
ANNOTATION_ROOT = Path("/kaggle/working/cct20-annotations")
STAGE_FILES = {
    "cis_validation": "cis_val_annotations.json",
    "trans_validation": "trans_val_annotations.json",
    "trans_test": "trans_test_annotations.json",
}


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def ensure_dependencies():
    run([sys.executable, "-m", "pip", "install", "-q", "open-clip-torch", "pillow"])


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


def find_named_file(filename):
    matches = sorted(Path("/kaggle/input").rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Attach the CCT-20 freeze artifacts containing {filename}.")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple attached files named {filename}; set up one unambiguous freeze dataset.")
    return matches[0]


def find_latest_ledger(stage):
    candidates = sorted({
        *Path("/kaggle/input").rglob("lesvi_cct20_ledger.json"),
        *Path("/kaggle/input").rglob("lesvi_cct20_next_ledger.json"),
    })
    eligible = []
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = payload.get("completed_stages", [])
        if stage == "cis_validation" and not completed:
            eligible.append((0, path))
        elif stage == "trans_validation" and completed == ["cis_validation"]:
            eligible.append((1, path))
        elif stage == "trans_test" and completed == ["cis_validation", "trans_validation"]:
            eligible.append((2, path))
    if len(eligible) != 1:
        raise FileNotFoundError(f"Expected exactly one immutable ledger eligible for {stage}; found {len(eligible)}.")
    return eligible[0][1]


def find_image_root(train_annotation):
    configured = os.environ.get("CCT20_IMAGE_ROOT")
    if configured and Path(configured).is_dir():
        return Path(configured)
    payload = json.loads(train_annotation.read_text(encoding="utf-8"))
    first_name = str(payload["images"][0]["file_name"])
    matches = sorted(Path("/kaggle/input").rglob(first_name))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Attach the official CCT-20 small benchmark images and ensure the first train image is unique, "
            "or set CCT20_IMAGE_ROOT."
        )
    return matches[0].parent


def main():
    stage = os.environ.get("LESVI_CCT_STAGE", "cis_validation")
    if stage not in STAGE_FILES:
        raise ValueError(f"Unsupported LESVI_CCT_STAGE: {stage!r}")
    clone_repository()
    ensure_dependencies()
    annotations = prepare_annotations()
    frozen_spec = find_named_file("lesvi_cct20_frozen_spec.json")
    prior = find_named_file("lesvi_cct20_prior.json")
    mapping = find_named_file("lesvi_cct20_class_mapping.json")
    manifest = find_named_file("lesvi_cct20_split_manifest.json")
    viability = find_named_file("lesvi_cct20_prior_viability.json")
    ledger = find_latest_ledger(stage)
    image_root = find_image_root(annotations / "train_annotations.json")
    output = Path(f"/kaggle/working/cct20-lesvi-{stage}")
    command = [
        sys.executable,
        "src/eval_lesvi_cct20.py",
        f"--stage={stage}",
        f"--train-annotations={annotations / 'train_annotations.json'}",
        f"--stage-annotations={annotations / STAGE_FILES[stage]}",
        f"--image-root={image_root}",
        f"--frozen-spec={frozen_spec}",
        f"--prior-artifact={prior}",
        f"--class-mapping={mapping}",
        f"--split-manifest={manifest}",
        f"--prior-viability={viability}",
        f"--ledger={ledger}",
        "--next-ledger=/kaggle/working/lesvi_cct20_next_ledger.json",
        f"--output-dir={output}",
        f"--workspace-root={WORKING_REPOSITORY}",
        "--batch-size=256",
        "--workers=2",
        "--device=auto",
    ]
    environment = dict(os.environ, PYTHONPATH=str(WORKING_REPOSITORY))
    run(command, cwd=WORKING_REPOSITORY, env=environment)


if __name__ == "__main__":
    main()
