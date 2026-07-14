import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
HARDCODED_WANDB_API_KEY = ""


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
    run([sys.executable, "kaggle_eval_drm_wise_tip_adapter.py"], cwd=WORKING_REPOSITORY, env=env)


if __name__ == "__main__":
    main()
