"""Environment creation and environment-specific metrics."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import gymnasium as gym
import numpy as np

from rl_analysis.config import EnvironmentProfile, get_env_profile


def make_env(env_id: str | EnvironmentProfile, *, seed: int | None = None, render_mode: str | None = None):
    profile = get_env_profile(env_id) if isinstance(env_id, str) else env_id
    if profile.env_id == "ALE/Freeway-v5":
        env = _make_freeway_env(profile, render_mode=render_mode)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
        return env
    kwargs: dict[str, Any] = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(profile.env_id, **kwargs)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def _register_ale() -> None:
    try:
        import ale_py
    except ImportError as exc:
        raise gym.error.DependencyNotInstalled(
            'Atari Freeway requires ale-py. Install with `uv pip install "gymnasium[atari]" ale-py`.'
        ) from exc

    register_envs = getattr(gym, "register_envs", None)
    if register_envs is not None:
        register_envs(ale_py)


def _make_freeway_env(profile: EnvironmentProfile, *, render_mode: str | None = None):
    from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward

    _register_ale()
    kwargs: dict[str, Any] = {
        "frameskip": profile.raw_frame_skip or 1,
        "repeat_action_probability": profile.repeat_action_probability
        if profile.repeat_action_probability is not None
        else 0.25,
        "full_action_space": bool(profile.full_action_space),
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    env = gym.make(profile.env_id, **kwargs)
    env = AtariPreprocessing(
        env,
        noop_max=profile.noop_max or 0,
        frame_skip=profile.frame_skip or 1,
        screen_size=profile.resize_shape or 84,
        terminal_on_life_loss=bool(profile.terminal_on_life_loss),
        grayscale_obs=bool(profile.grayscale),
        grayscale_newaxis=False,
        scale_obs=False,
    )
    if profile.reward_clipping:
        env = TransformReward(env, lambda reward: float(np.sign(float(reward))))
    if profile.frame_stack and profile.frame_stack > 1:
        env = FrameStackObservation(env, stack_size=profile.frame_stack)
    return env


def environment_config_dict(profile: EnvironmentProfile) -> dict[str, Any]:
    payload = asdict(profile)
    payload["resize_shape"] = list(profile.resize_shape) if profile.resize_shape is not None else None
    payload["score_thresholds"] = list(profile.score_thresholds)
    return payload


def lunarlander_episode_metrics(
    *,
    episode_return: float,
    episode_length: int,
    action_counts: np.ndarray,
    final_observation: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> dict[str, Any]:
    safe_length = max(1, episode_length)
    final = np.asarray(final_observation, dtype=np.float32)
    both_legs = bool(final[6] > 0.5 and final[7] > 0.5) if final.shape[0] >= 8 else False
    one_leg = bool((final[6] > 0.5) ^ (final[7] > 0.5)) if final.shape[0] >= 8 else False
    return {
        "landing_success": bool(episode_return >= 200.0),
        "crash": bool(terminated and episode_return < 0.0),
        "timeout": bool(truncated),
        "fuel_proxy_main_engine_count": int(action_counts[2]) if action_counts.size > 2 else 0,
        "fuel_proxy_side_engine_count": int(action_counts[1] + action_counts[3]) if action_counts.size > 3 else 0,
        "main_engine_action_fraction": float(action_counts[2] / safe_length) if action_counts.size > 2 else 0.0,
        "side_engine_action_fraction": float((action_counts[1] + action_counts[3]) / safe_length)
        if action_counts.size > 3
        else 0.0,
        "final_x_position": float(final[0]) if final.shape[0] > 0 else None,
        "final_y_position": float(final[1]) if final.shape[0] > 1 else None,
        "final_x_velocity": float(final[2]) if final.shape[0] > 2 else None,
        "final_y_velocity": float(final[3]) if final.shape[0] > 3 else None,
        "final_angle": float(final[4]) if final.shape[0] > 4 else None,
        "final_angular_velocity": float(final[5]) if final.shape[0] > 5 else None,
        "final_left_leg_contact": bool(final[6] > 0.5) if final.shape[0] > 6 else None,
        "final_right_leg_contact": bool(final[7] > 0.5) if final.shape[0] > 7 else None,
        "both_legs_contact": both_legs,
        "one_leg_contact": one_leg,
    }


def freeway_episode_metrics(
    *,
    episode_return: float,
    episode_length: int,
    action_counts: np.ndarray,
    max_score_so_far: float,
) -> dict[str, Any]:
    safe_length = max(1, episode_length)
    score = float(episode_return)
    return {
        "score": int(round(score)),
        "score_raw": score,
        "zero_score_episode": bool(score <= 0.0),
        "max_score_so_far": float(max_score_so_far),
        "action_noop_fraction": float(action_counts[0] / safe_length) if action_counts.size > 0 else 0.0,
        "action_up_fraction": float(action_counts[1] / safe_length) if action_counts.size > 1 else 0.0,
        "action_down_fraction": float(action_counts[2] / safe_length) if action_counts.size > 2 else 0.0,
    }
