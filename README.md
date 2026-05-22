# LunarLander and Freeway DQN Experiments

This repository contains a PyTorch/Gymnasium framework for DQN ablation experiments on `LunarLander-v3` and Atari Freeway.

## Installation

```bash
uv venv
uv pip install -r requirements.txt
```

## Basic Examples

Train DQN on LunarLander:

```bash
uv run python -m rl_analysis.train --env LunarLander-v3 --variant dqn --seed 0
```

Train DQN on Freeway:

```bash
uv run python -m rl_analysis.train --env freeway --variant dqn --seed 0
```

Record a demo from a trained checkpoint:

```bash
uv run python -m rl_analysis.demo \
  --checkpoint selected_checkpoints/freeway_agent.pt \
  --episodes 1 \
  --video-dir demos
```

Experiment outputs are written under `output/<env_slug>/<variant>/seed_<seed>/<run_id>/`.
