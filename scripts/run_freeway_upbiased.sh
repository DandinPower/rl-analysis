#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
source "$VENV_DIR/bin/activate"

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/run_freeway_upbiased_${RUN_STAMP}}"
SEEDS=(0 1 2 3)
VARIANTS=(dqn_no_replay)

echo "Writing experiment outputs to: $OUTPUT_ROOT"
echo "Running Freeway up-biased variants: ${VARIANTS[*]}"
echo "Using seeds: ${SEEDS[*]}"

for VARIANT in "${VARIANTS[@]}"; do
  echo "Starting variant: $VARIANT"
  python -m rl_analysis.train \
    --env freeway \
    --variant "$VARIANT" \
    --seeds "${SEEDS[@]}" \
    --max-seed-workers 4 \
    --output-root "$OUTPUT_ROOT" \
    --freeway-up-bias 0.5
done

python -m rl_analysis.aggregate \
  --env freeway \
  --output-root "$OUTPUT_ROOT"

echo "Done. Aggregates and plots are under: $OUTPUT_ROOT/freeway"
