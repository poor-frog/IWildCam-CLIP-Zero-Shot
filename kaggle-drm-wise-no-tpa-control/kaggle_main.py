import os
import subprocess
import sys


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = "/kaggle/working/IWildCam-CLIP-Zero-Shot"
WISE_ALPHA_GRID = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
HARDCODED_WANDB_API_KEY = ""


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def clone_or_update_repository():
    if os.path.exists(WORKING_REPOSITORY):
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def main():
    clone_or_update_repository()
    env = dict(os.environ)
    if HARDCODED_WANDB_API_KEY and not env.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
    env["DRM_WISE_ALPHA_GRID"] = WISE_ALPHA_GRID
    env["DRM_STMP_PROTOTYPE_SCALE_GRID"] = "0"
    env["DRM_STMP_SEQUENCE_CONSENSUS_GRID"] = "0"
    env["DRM_STP_MULTI_PROTOTYPE_K_GRID"] = "1"
    env["DRM_STMP_TAIL_GAMMA_GRID"] = "0"
    env["DRM_STMP_GATE_MODE_GRID"] = "none"
    env["DRM_STMP_GATE_STRENGTH_GRID"] = "0"
    env["DRM_WISE_WANDB_RUN_PREFIX"] = "drm-wise-no-tpa-control-vitb16-iwildcamval"
    run([sys.executable, "kaggle_eval_drm_stmp_adapter.py"], cwd=WORKING_REPOSITORY, env=env)


if __name__ == "__main__":
    main()
