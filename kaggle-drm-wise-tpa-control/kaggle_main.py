import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
DEFAULT_WISE_ALPHA_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"


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
    env.setdefault("DRM_WISE_ALPHA_GRID", DEFAULT_WISE_ALPHA_GRID)
    env["DRM_STMP_SEQUENCE_CONSENSUS_GRID"] = "0"
    env.setdefault("DRM_WISE_WANDB_RUN_PREFIX", "drm-wise-tpa-control-vitb16-iwildcamval")
    run([sys.executable, "kaggle_eval_drm_stmp_adapter.py"], cwd=WORKING_REPOSITORY, env=env)


if __name__ == "__main__":
    main()
