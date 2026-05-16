"""Command line entrypoint for training DQN experiments."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rl_analysis.config import VARIANT_NAMES, RunConfig, build_run_config
from rl_analysis.trainer import DQNTrainer


DEFAULT_SEEDS = [0, 1, 2, 3, 4]


@dataclass(frozen=True)
class TrainResult:
    run_id: str
    run_dir: Path
    variant: str
    seed: int
    final_eval_return_mean: Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN ablations on Gymnasium environments.")
    parser.add_argument("--env", default="LunarLander-v3", help="Environment id or alias.")
    parser.add_argument("--variant", default="dqn", choices=[*VARIANT_NAMES, "all"], help="Variant to train.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --variant is not all.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Seeds to run. Overrides --seed; defaults to 0 1 2 3 4 for --variant all.",
    )
    parser.add_argument("--output-root", default="output", help="Root directory for experiment artifacts.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:N.")
    parser.add_argument(
        "--max-seed-workers",
        type=_positive_int,
        default=None,
        help="Maximum seed processes to run concurrently for each variant. Defaults to all selected seeds.",
    )
    parser.add_argument("--serial", action="store_true", help="Run selected seeds one at a time.")

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
    return parser.parse_args(argv)


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


def _selected_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        return args.seeds
    if args.variant == "all":
        return list(DEFAULT_SEEDS)
    return [args.seed]


def _build_configs(args: argparse.Namespace, variant: str, seeds: list[int]) -> list[RunConfig]:
    return [
        build_run_config(
            env_id=args.env,
            variant_name=variant,
            seed=seed,
            output_root=Path(args.output_root),
            training_overrides=_training_overrides(args),
            dqn_overrides=_dqn_overrides(args),
            optimizer_overrides=_optimizer_overrides(args),
            exploration_overrides=_exploration_overrides(args),
        )
        for seed in seeds
    ]


def _run_config(config: RunConfig) -> TrainResult:
    summary = DQNTrainer(config).run()
    return TrainResult(
        run_id=config.run_id,
        run_dir=config.run_dir,
        variant=config.variant.name,
        seed=config.seed,
        final_eval_return_mean=summary["performance"]["final_eval_return_mean"],
    )


def _print_completed(result: TrainResult) -> None:
    print(
        f"Completed {result.run_id}: final_eval_return_mean={result.final_eval_return_mean}",
        flush=True,
    )


def _print_failed(config: RunConfig, exc: BaseException) -> None:
    print(f"Failed {config.run_id} (variant={config.variant.name}, seed={config.seed}): {exc}", file=sys.stderr)


def _resolve_max_workers(args: argparse.Namespace, num_configs: int) -> int:
    if num_configs <= 0:
        return 0
    if args.max_seed_workers is None:
        return num_configs
    return min(args.max_seed_workers, num_configs)


def _run_configs_serial(configs: list[RunConfig]) -> int:
    failures = 0
    for config in configs:
        print(f"Starting {config.run_id} -> {config.run_dir}", flush=True)
        try:
            _print_completed(_run_config(config))
        except Exception as exc:
            failures += 1
            _print_failed(config, exc)
    return failures


def _run_configs_parallel(configs: list[RunConfig], max_workers: int) -> int:
    failures = 0
    for config in configs:
        print(f"Starting {config.run_id} -> {config.run_dir}", flush=True)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_config = {executor.submit(_run_config, config): config for config in configs}
        for future in as_completed(future_to_config):
            config = future_to_config[future]
            try:
                _print_completed(future.result())
            except Exception as exc:
                failures += 1
                _print_failed(config, exc)
    return failures


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    variants = VARIANT_NAMES if args.variant == "all" else (args.variant,)
    seeds = _selected_seeds(args)
    total_failures = 0
    for variant in variants:
        configs = _build_configs(args, variant, seeds)
        max_workers = _resolve_max_workers(args, len(configs))
        if args.serial or max_workers == 1:
            total_failures += _run_configs_serial(configs)
        else:
            total_failures += _run_configs_parallel(configs, max_workers)
    if total_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
