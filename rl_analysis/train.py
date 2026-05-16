"""Command line entrypoint for training DQN experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_analysis.config import VARIANT_NAMES, build_run_config
from rl_analysis.trainer import DQNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN ablations on Gymnasium environments.")
    parser.add_argument("--env", default="LunarLander-v3", help="Environment id or alias.")
    parser.add_argument("--variant", default="dqn", choices=[*VARIANT_NAMES, "all"], help="Variant to train.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --variant is not all.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="Seeds for --variant all.")
    parser.add_argument("--output-root", default="output", help="Root directory for experiment artifacts.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")

    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--eval-frequency-env-steps", type=int, default=None)
    parser.add_argument("--checkpoint-frequency-env-steps", type=int, default=None)
    parser.add_argument("--num-eval-episodes", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--update-log-frequency", type=int, default=None)
    parser.add_argument("--system-metrics-frequency-env-steps", type=int, default=None)

    parser.add_argument("--learning-starts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-buffer-size", type=int, default=None)
    parser.add_argument("--train-frequency-env-steps", type=int, default=None)
    parser.add_argument("--target-update-frequency-env-steps", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--epsilon-decay-env-steps", type=int, default=None)
    parser.add_argument("--epsilon-final", type=float, default=None)
    return parser.parse_args()


def _training_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "total_env_steps": args.total_env_steps,
        "eval_frequency_env_steps": args.eval_frequency_env_steps,
        "checkpoint_frequency_env_steps": args.checkpoint_frequency_env_steps,
        "num_eval_episodes": args.num_eval_episodes,
        "max_episodes": args.max_episodes,
        "update_log_frequency": args.update_log_frequency,
        "system_metrics_frequency_env_steps": args.system_metrics_frequency_env_steps,
        "device": args.device,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _dqn_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "replay_buffer_size": args.replay_buffer_size,
        "train_frequency_env_steps": args.train_frequency_env_steps,
        "target_update_frequency_env_steps": args.target_update_frequency_env_steps,
        "gamma": args.gamma,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def _optimizer_overrides(args: argparse.Namespace) -> dict:
    mapping = {"learning_rate": args.learning_rate}
    return {key: value for key, value in mapping.items() if value is not None}


def _exploration_overrides(args: argparse.Namespace) -> dict:
    mapping = {
        "epsilon_decay_env_steps": args.epsilon_decay_env_steps,
        "epsilon_final": args.epsilon_final,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def main() -> None:
    args = parse_args()
    variants = VARIANT_NAMES if args.variant == "all" else (args.variant,)
    seeds = args.seeds if args.variant == "all" else [args.seed]
    for variant in variants:
        for seed in seeds:
            config = build_run_config(
                env_id=args.env,
                variant_name=variant,
                seed=seed,
                output_root=Path(args.output_root),
                training_overrides=_training_overrides(args),
                dqn_overrides=_dqn_overrides(args),
                optimizer_overrides=_optimizer_overrides(args),
                exploration_overrides=_exploration_overrides(args),
            )
            print(f"Starting {config.run_id} -> {config.run_dir}")
            summary = DQNTrainer(config).run()
            final_eval = summary["performance"]["final_eval_return_mean"]
            print(f"Completed {config.run_id}: final_eval_return_mean={final_eval}")


if __name__ == "__main__":
    main()
