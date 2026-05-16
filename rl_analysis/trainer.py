"""Training loop for LunarLander DQN ablations."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl_analysis.agent import DQNAgent
from rl_analysis.config import RunConfig
from rl_analysis.envs import environment_config_dict, lunarlander_episode_metrics, make_env
from rl_analysis.logging_utils import RunLoggers, collect_system_metrics, write_run_config
from rl_analysis.networks import build_q_network, network_architecture_dict
from rl_analysis.replay import ReplayBuffer, Transition, replay_diagnostics, transition_to_batch
from rl_analysis.utils import (
    action_entropy,
    count_parameters,
    git_info,
    hardware_info,
    numeric_stats,
    read_jsonl,
    resolve_device,
    rolling_mean,
    rolling_std,
    set_global_seeds,
    software_info,
    to_jsonable,
    write_json,
)


class DQNTrainer:
    def __init__(self, config: RunConfig, *, repo_root: Path | None = None):
        self.config = config
        self.repo_root = repo_root or Path.cwd()
        self.rng = np.random.default_rng(config.seed)
        self.eval_rng = np.random.default_rng(config.seed + 10_000)
        set_global_seeds(config.seed)
        self.device = resolve_device(config.training.device)

        self.env = make_env(config.env, seed=config.seed)
        self.eval_env = make_env(config.env, seed=config.seed + 1_000)
        observation_shape = tuple(int(x) for x in self.env.observation_space.shape)
        action_dim = int(self.env.action_space.n)

        online_network = build_q_network(observation_shape, action_dim, config.network)
        target_network = (
            build_q_network(observation_shape, action_dim, config.network) if config.variant.use_target_network else None
        )
        num_parameters = count_parameters(online_network)
        self.config.network = replace(self.config.network, num_parameters=num_parameters)
        self.agent = DQNAgent(
            online_network=online_network,
            target_network=target_network,
            variant=config.variant,
            dqn_config=config.dqn,
            optimizer_config=config.optimizer,
            device=self.device,
        )
        self.replay_buffer = (
            ReplayBuffer(config.dqn.replay_buffer_size, observation_shape) if config.variant.use_replay_buffer else None
        )
        self.run_dir = config.run_dir
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.eval_records: list[dict[str, Any]] = []
        self.update_records: list[dict[str, Any]] = []
        self.failure_flags = {
            "nan_detected": False,
            "inf_detected": False,
            "q_value_explosion": False,
            "loss_explosion": False,
            "gradient_explosion": False,
            "early_terminated": False,
        }
        self.wall_time_start = 0.0

    def run(self) -> dict[str, Any]:
        import time

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.status = "running"
        self._write_run_config()
        self.wall_time_start = time.perf_counter()
        updates_done = 0
        last_update_time = self.wall_time_start

        next_eval_step = self.config.training.eval_frequency_env_steps
        next_checkpoint_step = self.config.training.checkpoint_frequency_env_steps
        next_system_step = self.config.training.system_metrics_frequency_env_steps
        latest_eval: dict[str, Any] | None = None

        obs, _ = self.env.reset(seed=self.config.seed)
        episode_return = 0.0
        episode_length = 0
        episode_index = 0
        episode_start = time.perf_counter()
        episode_action_counts = np.zeros(self.agent.action_dim, dtype=np.int64)
        global_step = 0

        with RunLoggers(self.run_dir) as loggers:
            while global_step < self.config.training.total_env_steps:
                epsilon = self.config.exploration.epsilon_at(global_step)
                action = self.agent.select_action(obs, epsilon, self.rng)
                next_obs, reward, terminated, truncated, _info = self.env.step(action)
                global_step += 1
                episode_return += float(reward)
                episode_length += 1
                episode_action_counts[action] += 1

                transition = Transition(
                    state=np.asarray(obs, dtype=np.float32),
                    action=action,
                    reward=float(reward),
                    next_state=np.asarray(next_obs, dtype=np.float32),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    global_step=global_step,
                    episode_index=episode_index,
                )
                batch_for_update = None
                if self.replay_buffer is not None:
                    self.replay_buffer.add(transition)
                    if self._should_update(global_step):
                        for _ in range(self.config.dqn.gradient_steps_per_train):
                            batch_for_update = self.replay_buffer.sample(self.config.dqn.batch_size, self.rng)
                            update_metrics = self._update_and_log(
                                batch_for_update, global_step, loggers, updates_done
                            )
                            updates_done += 1
                            self._merge_failure_flags(update_metrics)
                elif self._should_update(global_step):
                    batch_for_update = transition_to_batch(transition)
                    update_metrics = self._update_and_log(batch_for_update, global_step, loggers, updates_done)
                    updates_done += 1
                    self._merge_failure_flags(update_metrics)

                self.agent.maybe_update_target(global_step)

                if terminated or truncated:
                    episode_wall_time = time.perf_counter() - episode_start
                    self.episode_returns.append(float(episode_return))
                    self.episode_lengths.append(int(episode_length))
                    game_specific = lunarlander_episode_metrics(
                        episode_return=episode_return,
                        episode_length=episode_length,
                        action_counts=episode_action_counts,
                        final_observation=np.asarray(next_obs, dtype=np.float32),
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                    )
                    loggers.train_episode.write(
                        {
                            "run_id": self.config.run_id,
                            "env_id": self.config.env.env_id,
                            "variant": self.config.variant.name,
                            "seed": self.config.seed,
                            "episode_index": episode_index,
                            "global_env_step": global_step,
                            "episode_return": float(episode_return),
                            "episode_length": episode_length,
                            "episode_wall_time_sec": episode_wall_time,
                            "epsilon": epsilon,
                            "learning_rate": self.config.optimizer.learning_rate,
                            "replay_buffer_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
                            "rolling_return_10": rolling_mean(self.episode_returns, 10),
                            "rolling_return_50": rolling_mean(self.episode_returns, 50),
                            "rolling_return_100": rolling_mean(self.episode_returns, 100),
                            "rolling_length_100": rolling_mean([float(x) for x in self.episode_lengths], 100),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "termination_reason": "truncated" if truncated else "environment",
                            "game_specific": game_specific,
                        }
                    )
                    episode_index += 1
                    if (
                        self.config.training.max_episodes is not None
                        and episode_index >= self.config.training.max_episodes
                    ):
                        break
                    obs, _ = self.env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    episode_start = time.perf_counter()
                    episode_action_counts = np.zeros(self.agent.action_dim, dtype=np.int64)
                else:
                    obs = next_obs

                if global_step >= next_eval_step or global_step >= self.config.training.total_env_steps:
                    latest_eval = self.evaluate(global_step, len(self.eval_records))
                    self.eval_records.append(latest_eval)
                    loggers.eval.write(latest_eval)
                    next_eval_step += self.config.training.eval_frequency_env_steps

                if global_step >= next_checkpoint_step or global_step >= self.config.training.total_env_steps:
                    checkpoint_metrics = self.save_checkpoint(global_step, latest_eval)
                    loggers.checkpoint.write(checkpoint_metrics)
                    next_checkpoint_step += self.config.training.checkpoint_frequency_env_steps

                if global_step >= next_system_step or global_step >= self.config.training.total_env_steps:
                    elapsed = time.perf_counter() - self.wall_time_start
                    since_update = max(1e-9, time.perf_counter() - last_update_time)
                    loggers.system.write(
                        collect_system_metrics(
                            run_id=self.config.run_id,
                            global_env_step=global_step,
                            wall_time_start=self.wall_time_start,
                            env_steps_per_second=global_step / elapsed if elapsed > 0 else None,
                            updates_per_second=updates_done / since_update if updates_done else 0.0,
                            replay_buffer_memory_gb=self.replay_buffer.memory_gb()
                            if self.replay_buffer is not None
                            else None,
                        )
                    )
                    next_system_step += self.config.training.system_metrics_frequency_env_steps

        summary = self.build_summary(global_step=global_step, total_updates=updates_done)
        write_json(self.run_dir / "summary.json", summary)
        self.config.status = "completed"
        self.config.datetime_end = datetime.now(timezone.utc).isoformat()
        self._write_run_config()
        self.env.close()
        self.eval_env.close()
        return summary

    def _should_update(self, global_step: int) -> bool:
        if global_step < self.config.dqn.learning_starts:
            return False
        if global_step % self.config.dqn.train_frequency_env_steps != 0:
            return False
        if self.replay_buffer is not None and len(self.replay_buffer) < self.config.dqn.batch_size:
            return False
        return True

    def _update_and_log(
        self, batch, global_step: int, loggers: RunLoggers, updates_done: int
    ) -> dict[str, Any]:
        update_metrics = self.agent.update(batch, global_step)
        update_metrics.update(
            {
                "run_id": self.config.run_id,
                "env_id": self.config.env.env_id,
                "variant": self.config.variant.name,
                "seed": self.config.seed,
                "replay": replay_diagnostics(
                    batch=batch,
                    replay_buffer=self.replay_buffer,
                    current_step=global_step,
                    replay_enabled=self.config.variant.use_replay_buffer,
                ),
            }
        )
        if update_metrics["update_index"] % self.config.training.update_log_frequency == 0:
            self.update_records.append(update_metrics)
            loggers.train_update.write(update_metrics)
        return update_metrics

    def _merge_failure_flags(self, update_metrics: dict[str, Any]) -> None:
        failures = update_metrics.get("failure_diagnostics", {})
        for key in self.failure_flags:
            self.failure_flags[key] = bool(self.failure_flags[key] or failures.get(key, False))

    def evaluate(self, global_step: int, eval_index: int) -> dict[str, Any]:
        returns = []
        lengths = []
        entropies = []
        action_counts_rows = []
        landing_successes = []
        crashes = []
        timeouts = []
        both_legs = []
        one_leg = []
        main_frac = []
        side_frac = []

        for episode in range(self.config.training.num_eval_episodes):
            obs, _ = self.eval_env.reset(seed=self.config.seed + 100_000 + eval_index * 1_000 + episode)
            done = False
            episode_return = 0.0
            episode_length = 0
            action_counts = np.zeros(self.agent.action_dim, dtype=np.int64)
            final_obs = obs
            terminated = False
            truncated = False
            while not done:
                action = self.agent.select_action(obs, self.config.exploration.eval_epsilon, self.eval_rng)
                obs, reward, terminated, truncated, _info = self.eval_env.step(action)
                final_obs = obs
                done = bool(terminated or truncated)
                episode_return += float(reward)
                episode_length += 1
                action_counts[action] += 1

            metrics = lunarlander_episode_metrics(
                episode_return=episode_return,
                episode_length=episode_length,
                action_counts=action_counts,
                final_observation=np.asarray(final_obs, dtype=np.float32),
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            returns.append(float(episode_return))
            lengths.append(float(episode_length))
            entropies.append(action_entropy(action_counts))
            action_counts_rows.append(action_counts)
            landing_successes.append(float(metrics["landing_success"]))
            crashes.append(float(metrics["crash"]))
            timeouts.append(float(metrics["timeout"]))
            both_legs.append(float(metrics["both_legs_contact"]))
            one_leg.append(float(metrics["one_leg_contact"]))
            main_frac.append(float(metrics["main_engine_action_fraction"]))
            side_frac.append(float(metrics["side_engine_action_fraction"]))

        return_stats = numeric_stats(returns)
        length_stats = numeric_stats(lengths)
        best_so_far = max([*self.eval_records, {"return_mean": -float("inf")}], key=lambda row: row["return_mean"])
        if return_stats["mean"] is not None and return_stats["mean"] >= best_so_far["return_mean"]:
            best_eval_return = return_stats["mean"]
            best_eval_step = global_step
        else:
            best_eval_return = best_so_far["return_mean"]
            best_eval_step = best_so_far["global_env_step"]

        action_counts_mean = np.mean(np.stack(action_counts_rows, axis=0), axis=0)
        return {
            "run_id": self.config.run_id,
            "env_id": self.config.env.env_id,
            "variant": self.config.variant.name,
            "seed": self.config.seed,
            "global_env_step": global_step,
            "eval_index": eval_index,
            "num_eval_episodes": self.config.training.num_eval_episodes,
            "eval_epsilon": self.config.exploration.eval_epsilon,
            "return_mean": return_stats["mean"],
            "return_std": return_stats["std"],
            "return_median": return_stats["median"],
            "return_min": return_stats["min"],
            "return_max": return_stats["max"],
            "return_p25": return_stats["p25"],
            "return_p75": return_stats["p75"],
            "episode_length_mean": length_stats["mean"],
            "episode_length_std": length_stats["std"],
            "best_eval_return_mean_so_far": best_eval_return,
            "best_eval_step_so_far": best_eval_step,
            "action_entropy_mean": float(np.mean(entropies)),
            "action_entropy_std": float(np.std(entropies)),
            "action_counts_mean": action_counts_mean.tolist(),
            "lunarlander_eval": {
                "success_rate": float(np.mean([1.0 if r >= 200.0 else 0.0 for r in returns])),
                "success_definition": "return >= 200",
                "landing_success_rate": float(np.mean(landing_successes)),
                "crash_rate": float(np.mean(crashes)),
                "timeout_rate": float(np.mean(timeouts)),
                "main_engine_action_fraction_mean": float(np.mean(main_frac)),
                "side_engine_action_fraction_mean": float(np.mean(side_frac)),
                "both_legs_contact_rate": float(np.mean(both_legs)),
                "one_leg_contact_rate": float(np.mean(one_leg)),
                "no_leg_contact_rate": float(1.0 - np.mean(both_legs) - np.mean(one_leg)),
            },
        }

    def save_checkpoint(self, global_step: int, latest_eval: dict[str, Any] | None) -> dict[str, Any]:
        checkpoint_path = self.checkpoint_dir / f"model_step_{global_step}.pt"
        payload = {
            "run_id": self.config.run_id,
            "global_env_step": global_step,
            "variant": self.config.variant.name,
            "seed": self.config.seed,
            "online_network_state_dict": self.agent.online_network.state_dict(),
            "target_network_state_dict": self.agent.target_network.state_dict()
            if self.agent.target_network is not None
            else None,
            "optimizer_state_dict": self.agent.optimizer.state_dict(),
            "config": to_jsonable(self.config),
        }
        torch.save(payload, checkpoint_path)
        model_size_mb = checkpoint_path.stat().st_size / 1024**2
        best_eval = None
        if latest_eval is not None:
            best_eval = latest_eval["return_mean"] == latest_eval["best_eval_return_mean_so_far"]
        return {
            "run_id": self.config.run_id,
            "global_env_step": global_step,
            "checkpoint_index": len(list(self.checkpoint_dir.glob("model_step_*.pt"))),
            "checkpoint_path": str(checkpoint_path.relative_to(self.run_dir)),
            "eval_return_mean_at_checkpoint": latest_eval.get("return_mean") if latest_eval else None,
            "eval_return_std_at_checkpoint": latest_eval.get("return_std") if latest_eval else None,
            "is_best_checkpoint_so_far": best_eval,
            "model_size_mb": float(model_size_mb),
            "optimizer_state_size_mb": None,
        }

    def build_summary(self, *, global_step: int, total_updates: int) -> dict[str, Any]:
        eval_returns = [float(row["return_mean"]) for row in self.eval_records if row.get("return_mean") is not None]
        eval_steps = [int(row["global_env_step"]) for row in self.eval_records]
        final_eval = self.eval_records[-1] if self.eval_records else None
        best_eval_return = max(eval_returns) if eval_returns else None
        best_eval_step = eval_steps[int(np.argmax(eval_returns))] if eval_returns else None
        auc = float(np.trapezoid(eval_returns, eval_steps)) if len(eval_returns) >= 2 else 0.0
        normalized_auc = auc / max(1, self.config.training.total_env_steps)
        first_threshold = self._first_threshold_step(eval_returns, eval_steps)
        sustained_threshold = self._first_sustained_threshold_step(eval_returns, eval_steps)
        collapse_count, largest_drop, largest_drop_fraction = self._collapse_metrics(eval_returns)
        wall_time_total = 0.0
        if self.wall_time_start:
            import time

            wall_time_total = time.perf_counter() - self.wall_time_start

        logged_update_rows = read_jsonl(self.run_dir / "train_update_metrics.jsonl")
        update_source = logged_update_rows or self.update_records
        last_updates = update_source[-max(1, len(update_source) // 10) :] if update_source else []
        q_max_values = [
            row.get("q_values", {}).get("online_q_max")
            for row in update_source
            if row.get("q_values", {}).get("online_q_max") is not None
        ]

        return {
            "run_id": self.config.run_id,
            "env_id": self.config.env.env_id,
            "variant": self.config.variant.name,
            "seed": self.config.seed,
            "training_completed": True,
            "total_env_steps": global_step,
            "total_episodes": len(self.episode_returns),
            "total_updates": total_updates,
            "performance": {
                "final_train_return_mean_last_100": rolling_mean(self.episode_returns, 100),
                "final_train_return_std_last_100": rolling_std(self.episode_returns, 100),
                "final_eval_return_mean": final_eval.get("return_mean") if final_eval else None,
                "final_eval_return_std": final_eval.get("return_std") if final_eval else None,
                "final_eval_return_median": final_eval.get("return_median") if final_eval else None,
                "best_eval_return_mean": best_eval_return,
                "best_eval_step": best_eval_step,
                "area_under_eval_curve": auc,
                "normalized_area_under_eval_curve": normalized_auc,
            },
            "sample_efficiency": {
                "threshold": self.config.env.success_threshold,
                "first_step_reaching_threshold": first_threshold,
                "sustained_threshold_window": self.config.training.sustained_threshold_window,
                "first_step_sustained_threshold": sustained_threshold,
                "never_reached_threshold": first_threshold is None,
            },
            "stability": {
                "rolling_return_std_mean": rolling_std(self.episode_returns, min(100, len(self.episode_returns)))
                if self.episode_returns
                else None,
                "rolling_return_std_max": self._rolling_return_std_max(),
                "eval_return_std_across_time": float(np.std(eval_returns)) if eval_returns else None,
                "catastrophic_collapse_count": collapse_count,
                "largest_eval_drop": largest_drop,
                "largest_eval_drop_fraction": largest_drop_fraction,
                "collapse_definition": "eval return mean drops by >= 30% from previous best after crossing threshold",
            },
            "optimization": self._optimization_summary(last_updates),
            "q_diagnostics": self._q_summary(last_updates, q_max_values),
            "replay_diagnostics": self._replay_summary(last_updates),
            "target_network_diagnostics": self._target_summary(last_updates),
            "compute": {
                "wall_time_total_sec": wall_time_total,
                "env_steps_per_second": global_step / wall_time_total if wall_time_total > 0 else None,
                "updates_per_second": total_updates / wall_time_total if wall_time_total > 0 else None,
            },
            "failure_diagnostics": dict(self.failure_flags),
        }

    def _first_threshold_step(self, eval_returns: list[float], eval_steps: list[int]) -> int | None:
        threshold = self.config.env.success_threshold
        if threshold is None:
            return None
        for value, step in zip(eval_returns, eval_steps, strict=False):
            if value >= threshold:
                return step
        return None

    def _first_sustained_threshold_step(self, eval_returns: list[float], eval_steps: list[int]) -> int | None:
        threshold = self.config.env.success_threshold
        if threshold is None:
            return None
        window = self.config.training.sustained_threshold_window
        for start in range(0, max(0, len(eval_returns) - window + 1)):
            chunk = eval_returns[start : start + window]
            if all(value >= threshold for value in chunk):
                return eval_steps[start]
        return None

    def _collapse_metrics(self, eval_returns: list[float]) -> tuple[int, float | None, float | None]:
        threshold = self.config.env.success_threshold
        if threshold is None or not eval_returns:
            return 0, None, None
        best = -float("inf")
        crossed = False
        count = 0
        largest_drop = 0.0
        largest_fraction = 0.0
        for value in eval_returns:
            if value >= threshold:
                crossed = True
            if crossed and best > 0:
                drop = max(0.0, best - value)
                fraction = drop / best if best else 0.0
                if fraction >= 0.30:
                    count += 1
                largest_drop = max(largest_drop, drop)
                largest_fraction = max(largest_fraction, fraction)
            best = max(best, value)
        return count, float(largest_drop), float(largest_fraction)

    def _rolling_return_std_max(self) -> float | None:
        if not self.episode_returns:
            return None
        window = min(100, len(self.episode_returns))
        values = [
            float(np.std(self.episode_returns[max(0, index - window + 1) : index + 1]))
            for index in range(len(self.episode_returns))
        ]
        return float(np.max(values))

    def _optimization_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "td_loss_mean_last_10pct": None,
                "td_error_abs_mean_last_10pct": None,
                "td_error_abs_p95_last_10pct": None,
                "grad_norm_mean_last_10pct": None,
                "grad_norm_max": None,
                "gradient_clip_fraction": None,
            }
        return {
            "td_loss_mean_last_10pct": float(np.mean([row["loss"]["td_loss_mean"] for row in rows])),
            "td_error_abs_mean_last_10pct": float(np.mean([row["loss"]["td_error_abs_mean"] for row in rows])),
            "td_error_abs_p95_last_10pct": float(np.mean([row["loss"]["td_error_abs_p95"] for row in rows])),
            "grad_norm_mean_last_10pct": float(np.mean([row["optimization"]["grad_norm"] for row in rows])),
            "grad_norm_max": float(np.max([row["optimization"]["grad_norm"] for row in rows])),
            "gradient_clip_fraction": float(np.mean([row["optimization"]["grad_norm_clipped"] for row in rows])),
        }

    def _q_summary(self, rows: list[dict[str, Any]], q_max_values: list[float]) -> dict[str, Any]:
        if not rows:
            return {
                "online_q_mean_last_10pct": None,
                "online_q_max_mean_last_10pct": None,
                "target_q_mean_last_10pct": None,
                "mean_q_overestimation_proxy_last_10pct": None,
                "max_q_value_seen": max(q_max_values) if q_max_values else None,
            }
        return {
            "online_q_mean_last_10pct": float(np.mean([row["q_values"]["online_q_mean"] for row in rows])),
            "online_q_max_mean_last_10pct": float(np.mean([row["q_values"]["online_q_max_mean"] for row in rows])),
            "target_q_mean_last_10pct": float(np.mean([row["q_values"]["target_q_mean"] for row in rows])),
            "mean_q_overestimation_proxy_last_10pct": float(
                np.mean([row["q_values"]["q_overestimation_proxy"] for row in rows])
            ),
            "max_q_value_seen": max(q_max_values) if q_max_values else None,
        }

    def _replay_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "replay_enabled": self.config.variant.use_replay_buffer,
                "final_buffer_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
                "sample_age_mean_last_10pct": None,
                "sample_consecutive_transition_fraction_last_10pct": None,
            }
        return {
            "replay_enabled": self.config.variant.use_replay_buffer,
            "final_buffer_size": len(self.replay_buffer) if self.replay_buffer is not None else 0,
            "sample_age_mean_last_10pct": float(
                np.mean([row["replay"]["sample_age_mean"] for row in rows if row["replay"]["sample_age_mean"] is not None])
            )
            if any(row["replay"]["sample_age_mean"] is not None for row in rows)
            else None,
            "sample_consecutive_transition_fraction_last_10pct": float(
                np.mean(
                    [
                        row["replay"]["sample_consecutive_transition_fraction"]
                        for row in rows
                        if row["replay"]["sample_consecutive_transition_fraction"] is not None
                    ]
                )
            )
            if any(row["replay"]["sample_consecutive_transition_fraction"] is not None for row in rows)
            else None,
        }

    def _target_summary(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "target_network_enabled": self.config.variant.use_target_network,
                "target_update_count": self.agent.target_update_count if self.agent.target_network is not None else None,
                "online_target_param_l2_mean": None,
                "online_target_param_l2_max": None,
            }
        values = [
            row["target_network"]["online_target_param_l2"]
            for row in rows
            if row["target_network"]["online_target_param_l2"] is not None
        ]
        return {
            "target_network_enabled": self.config.variant.use_target_network,
            "target_update_count": self.agent.target_update_count if self.agent.target_network is not None else None,
            "online_target_param_l2_mean": float(np.mean(values)) if values else None,
            "online_target_param_l2_max": float(np.max(values)) if values else None,
        }

    def _write_run_config(self) -> None:
        algorithm = {
            "base": "DQN",
            "use_target_network": self.config.variant.use_target_network,
            "use_replay_buffer": self.config.variant.use_replay_buffer,
            "use_double_dqn": self.config.variant.use_double_dqn,
            "use_dueling_network": self.config.variant.use_dueling_network,
            "use_prioritized_replay": False,
            "use_noisy_network": False,
            "use_reward_clipping": self.config.variant.use_reward_clipping,
        }
        payload = {
            "run_id": self.config.run_id,
            "experiment_group": self.config.experiment_group,
            "env_id": self.config.env.env_id,
            "variant": self.config.variant.name,
            "variant_label": self.config.variant.label,
            "seed": self.config.seed,
            "datetime_start": self.config.datetime_start,
            "datetime_end": self.config.datetime_end,
            "status": self.config.status,
            "algorithm": algorithm,
            "training_budget": {
                "total_env_steps": self.config.training.total_env_steps,
                "max_episodes": self.config.training.max_episodes,
                "max_episode_steps": self.config.env.max_episode_steps,
                "num_seeds": self.config.training.num_seeds,
                "eval_frequency_env_steps": self.config.training.eval_frequency_env_steps,
                "checkpoint_frequency_env_steps": self.config.training.checkpoint_frequency_env_steps,
            },
            "environment": environment_config_dict(self.config.env),
            "network": network_architecture_dict(self.config.network, self.config.network.num_parameters or 0),
            "optimizer": asdict(self.config.optimizer),
            "dqn_hyperparameters": asdict(self.config.dqn),
            "exploration": asdict(self.config.exploration),
            "software": software_info(),
            "hardware": hardware_info(self.device),
            "code": git_info(self.repo_root),
        }
        write_run_config(self.run_dir, payload)
