#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SESSION="${COLAB_SESSION:-poorfrogs-maple}"
GPU="${COLAB_GPU:-T4}"
REMOTE_ROOT="${COLAB_REMOTE_ROOT:-/content/PoorFrogs}"
ARCHIVE="${TMPDIR:-/tmp}/poorfrogs-colab.tar.gz"

if ! command -v colab >/dev/null 2>&1; then
  echo "Error: colab CLI not found. Install with: uv tool install google-colab-cli" >&2
  exit 1
fi

cd "$REPO_ROOT"

echo "Packing PoorFrogs project..."
tar -czf "$ARCHIVE" \
  --exclude=.git \
  --exclude=.venv \
  --exclude=__pycache__ \
  --exclude='*.pyc' \
  --exclude=data \
  --exclude=wandb \
  --exclude=checkpoints \
  .

SESSIONS_OUTPUT="$(colab sessions 2>/dev/null || true)"
if [[ "$SESSIONS_OUTPUT" == *"$SESSION"* ]]; then
  echo "Using existing Colab session '$SESSION'..."
else
  echo "Creating Colab session '$SESSION' with GPU '$GPU'..."
  colab new -s "$SESSION" --gpu "$GPU"
fi

echo "Uploading project archive to Colab..."
if ! UPLOAD_OUTPUT="$(colab upload -s "$SESSION" "$ARCHIVE" /content/poorfrogs-colab.tar.gz 2>&1)"; then
  if [[ "$UPLOAD_OUTPUT" == *"Session '$SESSION' not found"* ]]; then
    echo "Colab session '$SESSION' was stale. Recreating it..."
    colab new -s "$SESSION" --gpu "$GPU"
    colab upload -s "$SESSION" "$ARCHIVE" /content/poorfrogs-colab.tar.gz
  else
    echo "$UPLOAD_OUTPUT" >&2
    exit 1
  fi
else
  printf '%s\n' "$UPLOAD_OUTPUT"
fi

echo "Extracting project and installing dependencies on Colab..."
colab exec -s "$SESSION" <<PY
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

remote_root = Path("$REMOTE_ROOT")
archive = Path("/content/poorfrogs-colab.tar.gz")

if remote_root.exists():
    shutil.rmtree(remote_root)
remote_root.mkdir(parents=True, exist_ok=True)

with tarfile.open(archive) as tar:
    tar.extractall(remote_root)

packages = [
    "braceexpand",
    "ftfy",
    "h5py",
    "open-clip-torch",
    "pandas",
    "regex",
    "tqdm",
    "wandb",
    "webdataset",
    "wilds",
]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-e", str(remote_root)])
print(f"Ready: {remote_root}")
PY

WANDB_FLAGS=()
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "Forwarding WANDB_API_KEY to Colab and enabling W&B logging..."
  colab exec -s "$SESSION" <<PY
from pathlib import Path
Path("/content/.poorfrogs_wandb_env").write_text("WANDB_API_KEY=${WANDB_API_KEY}\n")
PY
  WANDB_FLAGS+=(--wandb --wandb-project=PoorFrogs --wandb-run-name=maple-vit-b16-bs256-colab)
else
  echo "WANDB_API_KEY is not set locally; running without W&B logging."
fi

WANDB_ARGS_JSON="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "${WANDB_FLAGS[@]}")"
USER_ARGS_JSON="$(python3 -c 'import json, sys; print(json.dumps(sys.argv[1:]))' "$@")"

echo "Starting MaPLe training on Colab..."
colab exec -s "$SESSION" <<PY
import json
import os
import subprocess
import sys
from pathlib import Path

remote_root = Path("$REMOTE_ROOT")
os.chdir(remote_root)
os.environ["PYTHONPATH"] = str(remote_root)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

wandb_env = Path("/content/.poorfrogs_wandb_env")
if wandb_env.exists():
    for line in wandb_env.read_text().splitlines():
        key, _, value = line.partition("=")
        if key and value:
            os.environ[key] = value

wandb_args = json.loads(r'''$WANDB_ARGS_JSON''')
user_args = json.loads(r'''$USER_ARGS_JSON''')

cmd = [
    sys.executable,
    "src/train_maple_full.py",
    "--model=ViT-B/16",
    "--train-dataset=IWildCam",
    "--eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD",
    "--data-location=/content/data",
    "--batch-size=256",
    "--workers=4",
    "--n-ctx=2",
    "--maple-prompt-depth=9",
    "--epochs=20",
    "--lr=1e-5",
    "--wd=0.2",
    "--val-dataset=IWildCamVal",
    "--best-metric=F1-macro_all",
    "--save=./checkpoints/maple_full_prompt_learner_vitb16_bs256_iwildcamval.pt",
    *wandb_args,
    *user_args,
]
print("Running:", " ".join(cmd))
subprocess.check_call(cmd)
PY

echo "Downloading checkpoints to ./checkpoints_colab..."
mkdir -p "$REPO_ROOT/checkpoints_colab"
colab download -s "$SESSION" "$REMOTE_ROOT/checkpoints" "$REPO_ROOT/checkpoints_colab" || true

echo "Done. Session '$SESSION' is still running. Stop it with: colab stop -s $SESSION"
