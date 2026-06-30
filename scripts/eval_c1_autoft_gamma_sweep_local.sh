#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

GAMMA_GRID="${GAMMA_GRID:-0,0.25,0.5,0.75,1.0,1.25}"
LOG_DIR="${LOG_DIR:-./scripts/logs/c1_autoft_gamma_sweep}"
RUN_TAU_GRID="${RUN_TAU_GRID:-0}"
RUN_EVAL_DATASETS="${RUN_EVAL_DATASETS:-IWildCamVal,IWildCamOOD}"
mkdir -p "$LOG_DIR"

IFS=',' read -r -a gammas <<< "$GAMMA_GRID"

for gamma in "${gammas[@]}"; do
  gamma="${gamma//[[:space:]]/}"
  if [[ -z "$gamma" ]]; then
    continue
  fi

  safe_gamma="${gamma//./p}"
  safe_gamma="${safe_gamma//-/m}"
  log_file="$LOG_DIR/gamma_${safe_gamma}.log"

  echo "=== Evaluating MAPLE_LORA_GAMMA=$gamma ===" | tee "$log_file"
  MAPLE_LORA_GAMMA="$gamma" \
  TAU_GRID="$RUN_TAU_GRID" \
  EVAL_DATASETS="$RUN_EVAL_DATASETS" \
  bash scripts/eval_c1_autoft_best_local.sh 2>&1 | tee -a "$log_file"
  echo "=== Finished MAPLE_LORA_GAMMA=$gamma; log: $log_file ===" | tee -a "$log_file"
  echo
done
