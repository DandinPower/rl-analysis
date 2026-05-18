```bash
uv pip install -r requirements.txt
uv pip install "gymnasium[box2d]"
uv pip install "gymnasium[atari,other]" ale-py opencv-python
```

# LunarLander and Freeway DQN Experiments

This repository contains a PyTorch/Gymnasium framework for DQN ablation
experiments. The first supported environment is `LunarLander-v3`; Atari Freeway
is supported with ALE preprocessing, frame stacking, CNN Q-networks, and
Freeway-specific score/action metrics.

Run all LunarLander variants with the reference-scale defaults:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant all --seeds 0 1 2 3 4
```

Seeds for each selected variant run concurrently by default. Use
`--max-seed-workers N` to cap concurrency, or `--serial` to run one seed at a
time.

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

Run the tuned Freeway learning check for the main variants:

```bash
./scripts/run_freeway_tuned_stage.sh stage1
```

This runs `dqn`, `double_dqn`, and `dueling_dqn` with a larger replay buffer,
longer replay warmup, slower epsilon decay, higher final epsilon, and two seed
workers. Stage 1 uses lighter diagnostics: 5 eval episodes every 50k env steps,
so each seed logs about 20 eval rows instead of 100. If the Stage 1 acceptance
check looks good, promote the same learning settings to a longer run:

```bash
./scripts/run_freeway_tuned_stage.sh stage2
```

Stage 2 uses 20 eval episodes every 100k env steps for report-quality curves
without returning to the very heavy 10k-step evaluation cadence.

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
