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
