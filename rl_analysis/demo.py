"""Run recorded demos from trained DQN checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo
from torch import nn

from rl_analysis.config import NetworkConfig, get_variant_spec
from rl_analysis.envs import make_env
from rl_analysis.networks import build_q_network
from rl_analysis.utils import resolve_device


REQUIRED_CHECKPOINT_KEYS = (
    "config",
    "online_network_state_dict",
    "run_id",
    "variant",
    "seed",
)


@dataclass(frozen=True)
class DemoResult:
    checkpoint_path: Path
    env_id: str
    run_id: str
    variant: str
    seed: int
    global_env_step: int | None
    episode_returns: list[float]
    episode_lengths: list[int]
    video_paths: list[Path]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a recorded policy demo from a DQN checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to a model_step_*.pt checkpoint.")
    parser.add_argument("--episodes", type=_positive_int, default=1, help="Number of episodes to record.")
    parser.add_argument("--video-dir", type=Path, default=Path("output/demos"), help="Directory for MP4 output.")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, cuda:N, or mps.")
    return parser.parse_args(argv)


def validate_checkpoint_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary.")

    missing = [key for key in REQUIRED_CHECKPOINT_KEYS if key not in payload]
    if missing:
        raise ValueError(f"Checkpoint payload is missing required key(s): {', '.join(missing)}.")

    config = payload["config"]
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary.")

    env_config = config.get("env")
    if not isinstance(env_config, dict) or not env_config.get("env_id"):
        raise ValueError("Checkpoint config must include config.env.env_id.")

    network_config = config.get("network")
    if not isinstance(network_config, dict):
        raise ValueError("Checkpoint config must include config.network.")

    if not isinstance(payload["online_network_state_dict"], dict):
        raise ValueError("Checkpoint online_network_state_dict must be a state-dict dictionary.")

    get_variant_spec(str(payload["variant"]))
    return payload


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    resolved_path = checkpoint_path.expanduser()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved_path}")
    payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    return validate_checkpoint_payload(payload)


def network_config_from_checkpoint(payload: dict[str, Any]) -> NetworkConfig:
    validate_checkpoint_payload(payload)
    allowed_fields = set(NetworkConfig.__dataclass_fields__)
    raw_config = payload["config"]["network"]
    return NetworkConfig(**{key: value for key, value in raw_config.items() if key in allowed_fields})


def env_id_from_checkpoint(payload: dict[str, Any]) -> str:
    validate_checkpoint_payload(payload)
    return str(payload["config"]["env"]["env_id"])


def build_network_from_checkpoint(
    payload: dict[str, Any],
    observation_shape: tuple[int, ...],
    action_dim: int,
    device: torch.device,
) -> nn.Module:
    network = build_q_network(observation_shape, action_dim, network_config_from_checkpoint(payload))
    network.load_state_dict(payload["online_network_state_dict"])
    network.to(device)
    network.eval()
    return network


def select_greedy_action(network: nn.Module, observation: np.ndarray, device: torch.device) -> int:
    obs_tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = network(obs_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def run_demo_episode(env, network: nn.Module, device: torch.device, *, seed: int) -> tuple[float, int]:
    observation, _info = env.reset(seed=seed)
    done = False
    episode_return = 0.0
    episode_length = 0
    while not done:
        action = select_greedy_action(network, observation, device)
        observation, reward, terminated, truncated, _info = env.step(action)
        episode_return += float(reward)
        episode_length += 1
        done = bool(terminated or truncated)
    return episode_return, episode_length


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return safe[:80] or "checkpoint"


def run_demo(
    *,
    checkpoint_path: Path,
    episodes: int,
    video_dir: Path,
    device_name: str,
) -> DemoResult:
    payload = load_checkpoint_payload(checkpoint_path)
    env_id = env_id_from_checkpoint(payload)
    run_id = str(payload["run_id"])
    variant = str(payload["variant"])
    seed = int(payload["seed"])
    global_env_step = payload.get("global_env_step")
    if global_env_step is not None:
        global_env_step = int(global_env_step)

    device = resolve_device(device_name)
    video_dir = video_dir.expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)
    existing_videos = set(video_dir.glob("*.mp4"))

    env = None
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    try:
        env = make_env(env_id, seed=seed, render_mode="rgb_array")
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda _episode_id: True,
            name_prefix=f"{_safe_name(run_id)}-demo",
            disable_logger=True,
        )
        observation_shape = tuple(int(dim) for dim in env.observation_space.shape)
        action_dim = int(env.action_space.n)
        network = build_network_from_checkpoint(payload, observation_shape, action_dim, device)

        for episode_index in range(episodes):
            episode_return, episode_length = run_demo_episode(
                env,
                network,
                device,
                seed=seed + 200_000 + episode_index,
            )
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
    finally:
        if env is not None:
            env.close()

    video_paths = sorted(path for path in video_dir.glob("*.mp4") if path not in existing_videos)
    return DemoResult(
        checkpoint_path=checkpoint_path.expanduser(),
        env_id=env_id,
        run_id=run_id,
        variant=variant,
        seed=seed,
        global_env_step=global_env_step,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        video_paths=video_paths,
    )


def print_summary(result: DemoResult) -> None:
    print(f"Checkpoint: {result.checkpoint_path}")
    print(f"Run: {result.run_id} ({result.env_id}, {result.variant}, seed={result.seed})")
    if result.global_env_step is not None:
        print(f"Checkpoint step: {result.global_env_step}")
    print(f"Episode returns: {result.episode_returns}")
    print(f"Episode lengths: {result.episode_lengths}")
    if result.video_paths:
        print("Videos:")
        for path in result.video_paths:
            print(f"  {path}")
    else:
        print("Videos: none found")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_demo(
        checkpoint_path=args.checkpoint,
        episodes=args.episodes,
        video_dir=args.video_dir,
        device_name=args.device,
    )
    print_summary(result)


if __name__ == "__main__":
    main()
