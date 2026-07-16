import os
import subprocess
import sys
from pathlib import Path


GITHUB_REPOSITORY = "https://github.com/poor-frog/IWildCam-CLIP-Zero-Shot.git"
WORKING_REPOSITORY = Path("/kaggle/working/IWildCam-CLIP-Zero-Shot")


def run(command, cwd=None, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=cwd, env=env)


def require_environment(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must point to a frozen confirmation artifact.")
    return value


def clone_or_update_repository():
    if WORKING_REPOSITORY.exists():
        run(["git", "-C", WORKING_REPOSITORY, "pull", "--ff-only"])
    else:
        run(["git", "clone", GITHUB_REPOSITORY, WORKING_REPOSITORY])


def main():
    frozen_spec = require_environment("STP_FROZEN_SPEC")
    ledger = require_environment("STP_CONFIRMATION_LEDGER")
    clone_or_update_repository()
    run([
        sys.executable,
        "src/eval_stp_internal_confirm.py",
        f"--frozen-spec={frozen_spec}",
        f"--confirmation-ledger={ledger}",
        f"--workspace-root={WORKING_REPOSITORY}",
        "--validate-only",
    ], cwd=WORKING_REPOSITORY, env=dict(os.environ))


if __name__ == "__main__":
    main()
