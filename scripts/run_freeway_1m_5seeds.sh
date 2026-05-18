#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="output/full_freeway_1m_5seeds_${RUN_STAMP}"
SEEDS=(0 1 2 3 4)

echo "Writing experiment outputs to: $OUTPUT_ROOT"
echo "Running all Freeway variants with seeds: ${SEEDS[*]}"

python -m rl_analysis.train \
  --env freeway \
  --variant all \
  --seeds "${SEEDS[@]}" \
  --max-seed-workers 2 \
  --output-root "$OUTPUT_ROOT"

python -m rl_analysis.aggregate \
  --env freeway \
  --output-root "$OUTPUT_ROOT"

echo "Done. Aggregates and plots are under: $OUTPUT_ROOT/freeway"
