import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")
AUDIT_ENVIRONMENT = {
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
    "DRM_LOO_BCPD_STRENGTH_GRID": "0",
    "DRM_STMP_BATCH_SIZE": "256",
    "DRM_STMP_WORKERS": "2",
    "DRM_STP_CANDIDATE_RELIABILITY_OUTPUT_DIR": "outputs/stp-candidate-reliability-audit-v0",
    "DRM_STP_CANDIDATE_RELIABILITY_BOOTSTRAP_SAMPLES": "2000",
    "DRM_STP_CANDIDATE_RELIABILITY_BOOTSTRAP_SEED": "20260721",
    "DRM_STP_CANDIDATE_RELIABILITY_SHUFFLE_COUNT": "20",
    "DRM_STP_CANDIDATE_RELIABILITY_SHUFFLE_SEED_START": "20260721",
    "DRM_STP_CANDIDATE_RELIABILITY_PERMUTATION_COUNT": "20",
    "DRM_STP_CANDIDATE_RELIABILITY_PERMUTATION_SEED_START": "20260741",
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
    environment.update(AUDIT_ENVIRONMENT)
    run([sys.executable, "kaggle_eval_drm_stmp_adapter.py"], cwd=WORKING_REPOSITORY, env=environment)


if __name__ == "__main__":
    main()
