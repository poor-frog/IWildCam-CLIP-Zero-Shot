#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

PACKAGES=(
  open_clip_torch
  torch
  torchvision
  wilds
  braceexpand
  webdataset
  h5py
  ftfy
  regex
  tqdm
  numpy
  wandb
)

if [[ ! -d "clip" ]]; then
  echo "Error: local OpenAI CLIP package directory not found: $SCRIPT_DIR/clip" >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv "$VENV_DIR" --python "$PYTHON_VERSION"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  uv pip install "${PACKAGES[@]}"
  uv pip install -e clip
else
  PYTHON_BIN="python3"
  if command -v "python$PYTHON_VERSION" >/dev/null 2>&1; then
    PYTHON_BIN="python$PYTHON_VERSION"
  fi

  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install "${PACKAGES[@]}"
  python -m pip install -e clip
fi

cat <<EOF

=== Setup complete ===

Activate:
  source $VENV_DIR/bin/activate

Run eval:
  KMP_DUPLICATE_LIB_OK=TRUE PYTHONPATH=. python src/main.py \\
    --model=ViT-B-16 \\
    --template=iwildcam_template \\
    --train-dataset=IWildCam \\
    --eval-datasets=IWildCamIDVal,IWildCamID,IWildCamOOD \\
    --data-location=/path/to/data/raw/

Optional W&B:
  wandb login
  Append: --wandb --wandb-project=PoorFrogs --wandb-run-name=vit-b-16-iwildcam

EOF
