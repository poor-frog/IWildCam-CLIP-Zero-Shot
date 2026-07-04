import os
import subprocess
import sys
from pathlib import Path

import kaggle_main


CHECKPOINT_NAME = "flyp_nodrm_wise_vitb16_iwildcamval_best.pt"


def find_checkpoint():
    candidates = [
        os.environ.get("FLYP_TPA_CHECKPOINT"),
        f"/kaggle/working/checkpoints/{CHECKPOINT_NAME}",
        f"/kaggle/working/IWildCam-CLIP-Zero-Shot/checkpoints/{CHECKPOINT_NAME}",
    ]
    for input_root in Path("/kaggle/input").glob("*"):
        candidates.append(str(input_root / CHECKPOINT_NAME))
        candidates.extend(str(path) for path in input_root.rglob(CHECKPOINT_NAME))

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    raise FileNotFoundError(
        f"Could not find {CHECKPOINT_NAME}. Set FLYP_TPA_CHECKPOINT to the checkpoint path "
        "or attach a Kaggle dataset containing this file."
    )


def main():
    repo_root = kaggle_main.ensure_repo_root()
    os.chdir(repo_root)
    kaggle_main.configure_import_path(repo_root)
    kaggle_main._ensure_deps()
    kaggle_main._ensure_local_package_installed(repo_root)
    kaggle_main.assert_cloned_repo_supports_runtime_flags(repo_root)
    kaggle_main._patch_iwildcam_val()
    data_location = kaggle_main.prepare_iwildcam_layout(repo_root)
    checkpoint = find_checkpoint()

    command = [
        sys.executable,
        "src/eval_tail_cache.py",
        "--model=ViT-B-16",
        "--train-dataset=IWildCam",
        "--val-dataset=IWildCamVal",
        "--eval-datasets=IWildCamIDVal,IWildCamVal,IWildCamID,IWildCamOOD",
        "--template=iwildcam_template",
        f"--data-location={data_location}",
        f"--load={checkpoint}",
        "--prototype-scale-grid=50",
        "--cache-tau-grid=0",
        "--max-cache-examples-per-class=0",
        "--batch-size=256",
        "--workers=4",
        "--device=auto",
    ]
    if kaggle_main._configure_wandb_from_kaggle_secret():
        command.extend([
            "--wandb",
            "--wandb-project=PoorFrogs",
            "--wandb-run-name=flyp-nodrm-wise-tail-prototype-adapter-scale50-tau0",
        ])
    else:
        command.append("--no-wandb")

    print("Running FLYP + Tail Prototype Adapter evaluation:")
    print(" ".join(command))
    subprocess.check_call(command)


if __name__ == "__main__":
    main()
