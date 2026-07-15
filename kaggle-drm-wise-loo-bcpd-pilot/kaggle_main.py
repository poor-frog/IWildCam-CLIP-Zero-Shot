import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
HARDCODED_WANDB_API_KEY = ""
PILOT_ENVIRONMENT = {
    "DRM_WISE_ALPHA_GRID": "",
    "DRM_WISE_EVAL_ALPHA": "0.2",
    "DRM_STMP_PROTOTYPE_SCALE_GRID": "50",
    "DRM_STMP_SEQUENCE_CONSENSUS_GRID": "0",
    "DRM_STP_MULTI_PROTOTYPE_K_GRID": "1",
    "DRM_STMP_MULTI_PROTOTYPE_REDUCTION": "max",
    "DRM_STMP_TAIL_GAMMA_GRID": "0",
    "DRM_STMP_GATE_MODE_GRID": "none",
    "DRM_STMP_GATE_STRENGTH_GRID": "0",
    "DRM_SCTR_STRENGTH_GRID": "0",
    "DRM_SCTR_TAIL_PROTECTION_GRID": "0",
    "DRM_LOO_BCPD_STRENGTH_GRID": "0,0.25,0.5,1",
    "DRM_LOO_BCPD_DIAGNOSTICS_REPORT": "/kaggle/working/loo_bcpd_diagnostics_iwildcamval.md",
    "DRM_LOO_BCPD_DIAGNOSTICS_SPLIT": "IWildCamVal",
    "DRM_SUMMARY_HEAD": "loo_bcpd",
    "DRM_STMP_BATCH_SIZE": "256",
    "DRM_STMP_WORKERS": "2",
    "DRM_SCTR_WANDB_RUN_NAME": "drm-wise-loo-bcpd-pilot-vitb16-iwildcamval",
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
    environment = dict(os.environ)
    environment.update(PILOT_ENVIRONMENT)
    if HARDCODED_WANDB_API_KEY:
        environment["WANDB_API_KEY"] = HARDCODED_WANDB_API_KEY
    run([sys.executable, "kaggle_eval_drm_stmp_adapter.py"], cwd=WORKING_REPOSITORY, env=environment)


if __name__ == "__main__":
    main()
