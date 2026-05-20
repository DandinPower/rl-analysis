#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR="${VENV_DIR:-.venv}"
source "$VENV_DIR/bin/activate"

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="output/full_lunarlander_10seeds_${RUN_STAMP}"
SEEDS=(0 1 2 3 4 5 6 7 8 9)

mkdir -p "$OUTPUT_ROOT"

echo "Writing experiment outputs to: $OUTPUT_ROOT"
echo "Running all variants with seeds: ${SEEDS[*]}"

python -m rl_analysis.train \
  --env LunarLander-v3 \
  --variant all \
  --seeds "${SEEDS[@]}" \
  --max-seed-workers 10 \
  --output-root "$OUTPUT_ROOT"

python -m rl_analysis.aggregate \
  --env LunarLander-v3 \
  --output-root "$OUTPUT_ROOT"

echo "Done. Aggregates and plots are under: $OUTPUT_ROOT/lunarlander_v3"
