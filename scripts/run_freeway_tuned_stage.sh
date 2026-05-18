#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source .venv/bin/activate

STAGE="${1:-stage1}"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
SEEDS=(0 1 2 3 4)
VARIANTS=(dqn double_dqn dueling_dqn)
MAX_SEED_WORKERS="${MAX_SEED_WORKERS:-2}"

case "$STAGE" in
  stage1)
    TOTAL_ENV_STEPS=1000000
    EVAL_FREQUENCY_ENV_STEPS=50000
    NUM_EVAL_EPISODES=5
    CHECKPOINT_FREQUENCY_ENV_STEPS=250000
    SYSTEM_METRICS_FREQUENCY_ENV_STEPS=50000
    OUTPUT_ROOT="output/freeway_tuned_stage1_${RUN_STAMP}"
    ;;
  stage2)
    TOTAL_ENV_STEPS=3000000
    EVAL_FREQUENCY_ENV_STEPS=100000
    NUM_EVAL_EPISODES=20
    CHECKPOINT_FREQUENCY_ENV_STEPS=500000
    SYSTEM_METRICS_FREQUENCY_ENV_STEPS=100000
    OUTPUT_ROOT="output/freeway_tuned_stage2_${RUN_STAMP}"
    ;;
  *)
    echo "Usage: $0 [stage1|stage2]" >&2
    exit 2
    ;;
esac

echo "Writing experiment outputs to: $OUTPUT_ROOT"
echo "Stage: $STAGE"
echo "Variants: ${VARIANTS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Max seed workers per variant: $MAX_SEED_WORKERS"
echo "Total env steps per run: $TOTAL_ENV_STEPS"
echo "Eval frequency: $EVAL_FREQUENCY_ENV_STEPS env steps"
echo "Eval episodes: $NUM_EVAL_EPISODES"
echo "Checkpoint frequency: $CHECKPOINT_FREQUENCY_ENV_STEPS env steps"

for variant in "${VARIANTS[@]}"; do
  echo
  echo "Running tuned Freeway variant: $variant"
  python -m rl_analysis.train \
    --env freeway \
    --variant "$variant" \
    --seeds "${SEEDS[@]}" \
    --max-seed-workers "$MAX_SEED_WORKERS" \
    --output-root "$OUTPUT_ROOT" \
    --total-env-steps "$TOTAL_ENV_STEPS" \
    --eval-frequency-env-steps "$EVAL_FREQUENCY_ENV_STEPS" \
    --num-eval-episodes "$NUM_EVAL_EPISODES" \
    --checkpoint-frequency-env-steps "$CHECKPOINT_FREQUENCY_ENV_STEPS" \
    --system-metrics-frequency-env-steps "$SYSTEM_METRICS_FREQUENCY_ENV_STEPS" \
    --learning-starts 100000 \
    --replay-buffer-size 100000 \
    --batch-size 32 \
    --train-frequency-env-steps 4 \
    --target-update-frequency-env-steps 10000 \
    --epsilon-final 0.10 \
    --epsilon-decay-env-steps 1000000 \
    --learning-rate 0.0001
done

python -m rl_analysis.aggregate \
  --env freeway \
  --output-root "$OUTPUT_ROOT"

echo
echo "Done. Aggregates and plots are under: $OUTPUT_ROOT/freeway"
