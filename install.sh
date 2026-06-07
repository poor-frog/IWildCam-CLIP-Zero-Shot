#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if [[ ! -f "pyproject.toml" ]]; then
  echo "Error: pyproject.toml not found in $SCRIPT_DIR" >&2
  exit 1
fi

if [[ ! -f "uv.lock" ]]; then
  echo "Error: uv.lock not found. Run 'uv lock' first to create the locked environment." >&2
  exit 1
fi

if [[ ! -d "clip" ]]; then
  echo "Error: local OpenAI CLIP package directory not found: $SCRIPT_DIR/clip" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  cat >&2 <<'EOF'
Error: uv is required for reproducible locked installs.
Install uv first, for example:
  curl -LsSf https://astral.sh/uv/install.sh | sh
EOF
  exit 1
fi

echo "Installing locked PoorFrogs environment into $VENV_DIR using Python $PYTHON_VERSION..."
UV_PROJECT_ENVIRONMENT="$VENV_DIR" uv sync --locked --python "$PYTHON_VERSION"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cat <<EOF

=== Setup complete ===

Environment was installed from uv.lock.

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
