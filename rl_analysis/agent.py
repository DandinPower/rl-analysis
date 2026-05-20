"""DQN agent implementation."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl_analysis.config import DQNHyperparameters, OptimizerConfig, VariantSpec
from rl_analysis.replay import TransitionBatch


def _as_tensor(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def _diagnostic_tensor(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.device.type == "mps":
        return tensor.float().cpu()
    return tensor


def calculate_dqn_targets(
    *,
    online_network: nn.Module,
    target_network: nn.Module,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    terminated: torch.Tensor,
    gamma: float,
    use_double_dqn: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        target_next_q = target_network(next_states)
        vanilla_max_target_q = target_next_q.max(dim=1).values
        if use_double_dqn:
            online_next_q = online_network(next_states)
            next_actions = online_next_q.argmax(dim=1, keepdim=True)
            selected_next_q = target_next_q.gather(1, next_actions).squeeze(1)
            online_selected_action_q = online_next_q.gather(1, next_actions).squeeze(1)
        else:
            online_next_q = online_network(next_states)
            selected_next_q = vanilla_max_target_q
            online_selected_action_q = online_next_q.max(dim=1).values

        non_terminal = 1.0 - terminated.float()
        targets = rewards + gamma * non_terminal * selected_next_q
        diagnostics = {
            "online_selected_action_q_mean": float(online_selected_action_q.mean().item()),
            "target_evaluated_selected_action_q_mean": float(selected_next_q.mean().item()),
            "target_evaluated_selected_action_q_std": float(selected_next_q.std(unbiased=False).item()),
            "target_evaluated_selected_action_q_min": float(selected_next_q.min().item()),
            "target_evaluated_selected_action_q_max": float(selected_next_q.max().item()),
            "vanilla_max_target_q_mean_proxy": float(vanilla_max_target_q.mean().item()),
            "double_dqn_target_q_mean": float(selected_next_q.mean().item()) if use_double_dqn else None,
            "estimated_overestimation_reduction": float((vanilla_max_target_q - selected_next_q).mean().item())
            if use_double_dqn
            else None,
        }
        return targets, diagnostics


class DQNAgent:
    def __init__(
        self,
        *,
        online_network: nn.Module,
        target_network: nn.Module | None,
        variant: VariantSpec,
        dqn_config: DQNHyperparameters,
        optimizer_config: OptimizerConfig,
        device: torch.device,
    ):
        self.online_network = online_network.to(device)
        self.target_network = target_network.to(device) if target_network is not None else None
        self.variant = variant
        self.dqn_config = dqn_config
        self.optimizer_config = optimizer_config
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=optimizer_config.learning_rate,
            eps=optimizer_config.epsilon,
            weight_decay=optimizer_config.weight_decay,
        )
        self.update_index = 0
        self.target_update_count = 0
        self.last_target_update_step = 0
        if self.target_network is not None:
            self.hard_update_target()

    def select_action(self, observation: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.action_dim))
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    @property
    def action_dim(self) -> int:
        return int(getattr(self.online_network, "action_dim"))

    def hard_update_target(self, global_env_step: int = 0) -> None:
        if self.target_network is None:
            return
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_update_count += 1
        self.last_target_update_step = global_env_step

    def maybe_update_target(self, global_env_step: int) -> bool:
        if self.target_network is None:
            return False
        frequency = self.dqn_config.target_update_frequency_env_steps
        if frequency <= 0 or global_env_step <= 0 or global_env_step % frequency != 0:
            return False
        tau = self.dqn_config.target_update_tau
        if tau >= 1.0:
            self.hard_update_target(global_env_step)
        else:
            with torch.no_grad():
                for target_param, online_param in zip(
                    self.target_network.parameters(), self.online_network.parameters(), strict=True
                ):
                    target_param.data.mul_(1.0 - tau).add_(online_param.data, alpha=tau)
            self.target_update_count += 1
            self.last_target_update_step = global_env_step
        return True

    def target_parameter_distance(self) -> tuple[float | None, float | None]:
        if self.target_network is None:
            return None, None
        sq_sum = 0.0
        count = 0
        with torch.no_grad():
            for online_param, target_param in zip(
                self.online_network.parameters(), self.target_network.parameters(), strict=True
            ):
                diff = (online_param - target_param).detach()
                sq_sum += float(torch.sum(diff * diff).item())
                count += diff.numel()
        l2 = math.sqrt(sq_sum)
        mean_per_param = l2 / max(1, count)
        return l2, mean_per_param

    def update(self, batch: TransitionBatch, global_env_step: int) -> dict[str, Any]:
        start = time.perf_counter()
        states = _as_tensor(batch.states, self.device, torch.float32)
        actions = _as_tensor(batch.actions, self.device, torch.int64)
        rewards = _as_tensor(batch.rewards, self.device, torch.float32)
        next_states = _as_tensor(batch.next_states, self.device, torch.float32)
        terminated = _as_tensor(batch.terminated.astype(np.float32), self.device, torch.float32)

        target_network = self.target_network if self.target_network is not None else self.online_network
        targets, double_dqn_diag = calculate_dqn_targets(
            online_network=self.online_network,
            target_network=target_network,
            rewards=rewards,
            next_states=next_states,
            terminated=terminated,
            gamma=self.dqn_config.gamma,
            use_double_dqn=self.variant.use_double_dqn,
        )

        q_values, streams = self.online_network(states, return_streams=True)
        chosen_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = targets - chosen_q
        per_sample_loss = F.smooth_l1_loss(chosen_q, targets, reduction="none")
        loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_before = torch.nn.utils.clip_grad_norm_(
            self.online_network.parameters(), self.dqn_config.max_grad_norm
        )
        self.optimizer.step()
        self.update_index += 1

        grad_norm_after = self._grad_norm()
        param_norm = self._param_norm()
        top2 = torch.topk(q_values.detach(), k=min(2, q_values.shape[1]), dim=1).values
        action_gap = top2[:, 0] - top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0])
        target_param_l2, target_param_l2_mean = self.target_parameter_distance()

        td_abs = torch.abs(td_errors.detach())
        td_abs_stats = _diagnostic_tensor(td_abs)
        target_q_mean = double_dqn_diag["target_evaluated_selected_action_q_mean"]
        online_q_max_mean = float(q_values.detach().max(dim=1).values.mean().item())
        metrics = {
            "global_env_step": global_env_step,
            "update_index": self.update_index,
            "loss": {
                "td_loss_mean": float(loss.detach().item()),
                "td_loss_std": float(per_sample_loss.detach().std(unbiased=False).item()),
                "td_error_mean": float(td_errors.detach().mean().item()),
                "td_error_abs_mean": float(td_abs_stats.mean().item()),
                "td_error_abs_median": float(torch.median(td_abs_stats).item()),
                "td_error_abs_p90": float(torch.quantile(td_abs_stats, 0.90).item()),
                "td_error_abs_p95": float(torch.quantile(td_abs_stats, 0.95).item()),
                "td_error_abs_p99": float(torch.quantile(td_abs_stats, 0.99).item()),
            },
            "q_values": {
                "online_q_mean": float(q_values.detach().mean().item()),
                "online_q_std": float(q_values.detach().std(unbiased=False).item()),
                "online_q_min": float(q_values.detach().min().item()),
                "online_q_max": float(q_values.detach().max().item()),
                "online_q_max_mean": online_q_max_mean,
                "target_q_mean": target_q_mean,
                "target_q_std": double_dqn_diag["target_evaluated_selected_action_q_std"],
                "target_q_min": double_dqn_diag["target_evaluated_selected_action_q_min"],
                "target_q_max": double_dqn_diag["target_evaluated_selected_action_q_max"],
                "q_overestimation_proxy": float(online_q_max_mean - target_q_mean),
                "action_gap_mean": float(action_gap.mean().item()),
                "action_gap_std": float(action_gap.std(unbiased=False).item()),
            },
            "targets": {
                "bellman_target_mean": float(targets.detach().mean().item()),
                "bellman_target_std": float(targets.detach().std(unbiased=False).item()),
                "bellman_target_min": float(targets.detach().min().item()),
                "bellman_target_max": float(targets.detach().max().item()),
                "reward_mean": float(rewards.mean().item()),
                "reward_std": float(rewards.std(unbiased=False).item()),
                "done_fraction": float(np.mean(batch.done_for_logging)),
            },
            "optimization": {
                "learning_rate": self.optimizer_config.learning_rate,
                "grad_norm": float(grad_norm_after),
                "grad_norm_before_clip": float(grad_norm_before.item()),
                "grad_norm_after_clip": float(grad_norm_after),
                "grad_norm_clipped": bool(float(grad_norm_before.item()) > self.dqn_config.max_grad_norm),
                "param_norm": float(param_norm),
                "update_wall_time_sec": time.perf_counter() - start,
            },
            "target_network": {
                "enabled": self.target_network is not None,
                "target_update_count": self.target_update_count if self.target_network is not None else None,
                "steps_since_target_update": global_env_step - self.last_target_update_step
                if self.target_network is not None
                else None,
                "target_update_frequency_env_steps": self.dqn_config.target_update_frequency_env_steps
                if self.target_network is not None
                else None,
                "target_update_tau": self.dqn_config.target_update_tau if self.target_network is not None else None,
                "online_target_param_l2": target_param_l2,
                "online_target_param_l2_mean_per_param": target_param_l2_mean,
            },
            "double_dqn": {
                "enabled": self.variant.use_double_dqn,
                **double_dqn_diag,
            },
            "dueling": self._dueling_metrics(streams, action_gap),
            "failure_diagnostics": self._failure_diagnostics(loss, q_values, grad_norm_before),
        }
        return metrics

    def _dueling_metrics(self, streams: dict[str, torch.Tensor | None], action_gap: torch.Tensor) -> dict[str, Any]:
        if not self.variant.use_dueling_network:
            return {
                "enabled": False,
                "value_stream_mean": None,
                "value_stream_std": None,
                "advantage_stream_mean": None,
                "advantage_stream_std": None,
                "advantage_abs_mean": None,
                "advantage_max_mean": None,
                "advantage_min_mean": None,
                "action_gap_mean": None,
                "action_gap_std": None,
            }
        value = streams["value"].detach()
        advantage = streams["advantage"].detach()
        return {
            "enabled": True,
            "value_stream_mean": float(value.mean().item()),
            "value_stream_std": float(value.std(unbiased=False).item()),
            "advantage_stream_mean": float(advantage.mean().item()),
            "advantage_stream_std": float(advantage.std(unbiased=False).item()),
            "advantage_abs_mean": float(torch.abs(advantage).mean().item()),
            "advantage_max_mean": float(advantage.max(dim=1).values.mean().item()),
            "advantage_min_mean": float(advantage.min(dim=1).values.mean().item()),
            "action_gap_mean": float(action_gap.mean().item()),
            "action_gap_std": float(action_gap.std(unbiased=False).item()),
        }

    def _grad_norm(self) -> float:
        sq_sum = 0.0
        for param in self.online_network.parameters():
            if param.grad is not None:
                sq_sum += float(torch.sum(param.grad.detach() * param.grad.detach()).item())
        return math.sqrt(sq_sum)

    def _param_norm(self) -> float:
        sq_sum = 0.0
        with torch.no_grad():
            for param in self.online_network.parameters():
                sq_sum += float(torch.sum(param * param).item())
        return math.sqrt(sq_sum)

    def _failure_diagnostics(
        self, loss: torch.Tensor, q_values: torch.Tensor, grad_norm_before: torch.Tensor
    ) -> dict[str, Any]:
        q_abs_max = float(torch.max(torch.abs(q_values.detach())).item())
        loss_value = float(loss.detach().item())
        grad_value = float(grad_norm_before.item())
        return {
            "nan_detected": bool(torch.isnan(loss).item() or torch.isnan(q_values).any().item()),
            "inf_detected": bool(torch.isinf(loss).item() or torch.isinf(q_values).any().item()),
            "q_value_explosion": bool(q_abs_max > 1000.0),
            "q_value_explosion_threshold": 1000.0,
            "loss_explosion": bool(loss_value > 1000.0),
            "loss_explosion_threshold": 1000.0,
            "gradient_explosion": bool(grad_value > 100.0),
            "gradient_explosion_threshold": 100.0,
            "early_terminated": False,
            "early_terminated_reason": None,
        }
