# LunarLander and Freeway DQN Experiments

This repository contains a PyTorch/Gymnasium framework for DQN ablation
experiments. The first supported environment is `LunarLander-v3`; Atari Freeway
is supported with ALE preprocessing, frame stacking, CNN Q-networks, and
Freeway-specific score/action metrics.

## Installation

```bash
uv pip install -r requirements.txt```

Run all LunarLander variants with the reference-scale defaults:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant all --seeds 0 1 2 3 4
```

Seeds for each selected variant run concurrently by default. Use
`--max-seed-workers N` to cap concurrency, or `--serial` to run one seed at a
time.

On Apple Silicon, use the MPS backend explicitly for local M1/M2/M3/M4 runs:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant dqn --seed 0 \
  --device mps \
  --serial
```

For multi-seed Apple Silicon runs, keep concurrency low so the unified memory
pool does not get oversubscribed:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant all --seeds 0 1 2 3 4 \
  --device mps \
  --max-seed-workers 1
```

Run the full 10-seed LunarLander experiment suite:

```bash
./scripts/run_full_lunarlander_10seeds.sh
```

This script writes to a fresh timestamped directory under
`output/full_lunarlander_10seeds_<timestamp>/`, so it does not overwrite earlier
experiment outputs.

Run the full 1M-step, 5-seed Freeway experiment suite:

```bash
./scripts/run_freeway_1m_5seeds.sh
```

This script caps seed concurrency at two workers because Atari image replay
buffers use much more memory than LunarLander vector replay buffers.

Run one small smoke experiment:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant dqn --seed 0 \
  --total-env-steps 200 \
  --eval-frequency-env-steps 100 \
  --checkpoint-frequency-env-steps 200 \
  --num-eval-episodes 1
```

Run one small Freeway smoke experiment:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env freeway --variant dqn --seed 0 \
  --total-env-steps 200 \
  --learning-starts 8 \
  --batch-size 2 \
  --replay-buffer-size 32 \
  --train-frequency-env-steps 4 \
  --eval-frequency-env-steps 100 \
  --checkpoint-frequency-env-steps 200 \
  --num-eval-episodes 1 \
```

Aggregate completed runs:

```bash
source .venv/bin/activate
python -m rl_analysis.aggregate --env LunarLander-v3
python -m rl_analysis.aggregate --env freeway
```

Experiment outputs are written under `output/<env_slug>/<variant>/seed_<seed>/<run_id>/`.
Each run contains `run_config.json`, JSONL metric files, checkpoints, and
`summary.json`.

Generate a source-code digest for the report appendix:

```bash
source .venv/bin/activate
gitingest rl_analysis -i "*.py" -o digest.txt
```
