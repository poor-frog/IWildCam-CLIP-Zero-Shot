import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
HARDCODED_WANDB_API_KEY = ""
BALANCED_TIP_ENVIRONMENT = {
    "DRM_TIP_ADAPTER_SUPPORT_SHOTS_GRID": "1,4,16",
    "DRM_TIP_ADAPTER_ALPHA_GRID": "0.000001,0.000003,0.00001,0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03",
    "DRM_TIP_ADAPTER_WANDB_RUN_NAME": "drm-wise-tip-adapter-balanced-k1-4-16-vitb16-iwildcamval",
}


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_or_update_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def main():
    clone_or_update_repository()
    env = dict(os.environ)
    if HARDCODED_WANDB_API_KEY and not env.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
    for name, value in BALANCED_TIP_ENVIRONMENT.items():
        env.setdefault(name, value)
    run([sys.executable, "kaggle_eval_drm_wise_tip_adapter.py"], cwd=WORKING_REPOSITORY, env=env)


if __name__ == "__main__":
    main()
