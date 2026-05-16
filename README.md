```bash
uv pip install -r requirements.txt
uv pip install "gymnasium[box2d]"
```

# LunarLander DQN Experiments

This repository contains a PyTorch/Gymnasium framework for DQN ablation
experiments. The first supported environment is `LunarLander-v3`; Atari Freeway
is represented as a future environment profile but is not runnable until ALE
support is added.

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

Run one small smoke experiment:

```bash
source .venv/bin/activate
python -m rl_analysis.train --env LunarLander-v3 --variant dqn --seed 0 \
  --total-env-steps 200 \
  --eval-frequency-env-steps 100 \
  --checkpoint-frequency-env-steps 200 \
  --num-eval-episodes 1
```

Aggregate completed runs:

```bash
source .venv/bin/activate
python -m rl_analysis.aggregate --env LunarLander-v3
```

Experiment outputs are written under `output/lunarlander_v3/<variant>/seed_<seed>/<run_id>/`.
Each run contains `run_config.json`, JSONL metric files, checkpoints, and
`summary.json`.

Generate a source-code digest for the report appendix:

```bash
source .venv/bin/activate
gitingest rl_analysis -i "*.py" -o digest.txt
```
