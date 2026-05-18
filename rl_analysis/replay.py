"""Replay-buffer data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    global_step: int
    episode_index: int


@dataclass
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    steps: np.ndarray
    episode_indices: np.ndarray

    @property
    def done_for_logging(self) -> np.ndarray:
        return np.logical_or(self.terminated, self.truncated)


def transition_to_batch(transition: Transition) -> TransitionBatch:
    return TransitionBatch(
        states=np.expand_dims(np.asarray(transition.state, dtype=np.float32), axis=0),
        actions=np.asarray([transition.action], dtype=np.int64),
        rewards=np.asarray([transition.reward], dtype=np.float32),
        next_states=np.expand_dims(np.asarray(transition.next_state, dtype=np.float32), axis=0),
        terminated=np.asarray([transition.terminated], dtype=np.bool_),
        truncated=np.asarray([transition.truncated], dtype=np.bool_),
        steps=np.asarray([transition.global_step], dtype=np.int64),
        episode_indices=np.asarray([transition.episode_index], dtype=np.int64),
    )


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        observation_dtype: np.dtype | type | str = np.float32,
    ):
        if capacity <= 0:
            raise ValueError("ReplayBuffer capacity must be positive.")
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        self.observation_dtype = np.dtype(observation_dtype)
        self.states = np.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.next_states = np.zeros((self.capacity, *self.observation_shape), dtype=self.observation_dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.steps = np.zeros(self.capacity, dtype=np.int64)
        self.episode_indices = np.zeros(self.capacity, dtype=np.int64)
        self._pos = 0
        self._size = 0

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        idx = self._pos
        self.states[idx] = np.asarray(transition.state, dtype=self.observation_dtype)
        self.next_states[idx] = np.asarray(transition.next_state, dtype=self.observation_dtype)
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.terminated[idx] = transition.terminated
        self.truncated[idx] = transition.truncated
        self.steps[idx] = transition.global_step
        self.episode_indices[idx] = transition.episode_index

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> TransitionBatch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        if batch_size > self._size:
            raise ValueError(f"Requested batch_size={batch_size}, but buffer only has {self._size} transitions.")
        indices = rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            states=self.states[indices].copy(),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=self.next_states[indices].copy(),
            terminated=self.terminated[indices].copy(),
            truncated=self.truncated[indices].copy(),
            steps=self.steps[indices].copy(),
            episode_indices=self.episode_indices[indices].copy(),
        )

    def memory_gb(self) -> float:
        total_bytes = (
            self.states.nbytes
            + self.next_states.nbytes
            + self.actions.nbytes
            + self.rewards.nbytes
            + self.terminated.nbytes
            + self.truncated.nbytes
            + self.steps.nbytes
            + self.episode_indices.nbytes
        )
        return float(total_bytes / 1024**3)


def replay_diagnostics(
    *,
    batch: TransitionBatch | None,
    replay_buffer: ReplayBuffer | None,
    current_step: int,
    replay_enabled: bool,
) -> dict[str, Any]:
    if not replay_enabled:
        return {
            "enabled": False,
            "buffer_size": None,
            "buffer_capacity": 0,
            "buffer_fill_fraction": None,
            "sample_age_mean": 0.0,
            "sample_age_std": 0.0,
            "sample_age_min": 0,
            "sample_age_max": 0,
            "sample_age_p50": 0.0,
            "sample_age_p90": 0.0,
            "sample_reward_mean": float(np.mean(batch.rewards)) if batch is not None else None,
            "sample_reward_std": float(np.std(batch.rewards)) if batch is not None else None,
            "sample_done_fraction": float(np.mean(batch.done_for_logging)) if batch is not None else None,
            "sample_unique_episode_count": 1 if batch is not None else None,
            "sample_consecutive_transition_fraction": 1.0 if batch is not None else None,
        }

    size = len(replay_buffer) if replay_buffer is not None else 0
    capacity = replay_buffer.capacity if replay_buffer is not None else 0
    if batch is None:
        return {
            "enabled": True,
            "buffer_size": size,
            "buffer_capacity": capacity,
            "buffer_fill_fraction": float(size / capacity) if capacity else None,
            "sample_age_mean": None,
            "sample_age_std": None,
            "sample_age_min": None,
            "sample_age_max": None,
            "sample_age_p50": None,
            "sample_age_p90": None,
            "sample_reward_mean": None,
            "sample_reward_std": None,
            "sample_done_fraction": None,
            "sample_unique_episode_count": None,
            "sample_consecutive_transition_fraction": None,
        }

    ages = np.maximum(0, current_step - batch.steps)
    sorted_steps = np.sort(batch.steps)
    consecutive = np.diff(sorted_steps) == 1
    return {
        "enabled": True,
        "buffer_size": size,
        "buffer_capacity": capacity,
        "buffer_fill_fraction": float(size / capacity) if capacity else None,
        "sample_age_mean": float(np.mean(ages)),
        "sample_age_std": float(np.std(ages)),
        "sample_age_min": int(np.min(ages)),
        "sample_age_max": int(np.max(ages)),
        "sample_age_p50": float(np.percentile(ages, 50)),
        "sample_age_p90": float(np.percentile(ages, 90)),
        "sample_reward_mean": float(np.mean(batch.rewards)),
        "sample_reward_std": float(np.std(batch.rewards)),
        "sample_done_fraction": float(np.mean(batch.done_for_logging)),
        "sample_unique_episode_count": int(np.unique(batch.episode_indices).size),
        "sample_consecutive_transition_fraction": float(np.mean(consecutive)) if consecutive.size else 0.0,
    }
