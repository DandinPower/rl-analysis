#!/usr/bin/env python3
"""Generate checkpoint-driven task-selection snapshots for the report."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rl-analysis-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rl_analysis.demo import build_network_from_checkpoint, load_checkpoint_payload, select_greedy_action
from rl_analysis.envs import make_env


DEFAULT_LUNARLANDER_CHECKPOINT = Path("selected_checkpoints/lunarlander_agent.pt")
DEFAULT_FREEWAY_CHECKPOINT = Path("selected_checkpoints/freeway_agent.pt")
DEFAULT_OUTPUT = Path("report/helper/figures/task_selection_snapshots.png")
DEFAULT_DEMO_SEED_OFFSET = 200_000


@dataclass(frozen=True)
class SnapshotRun:
    title: str
    checkpoint_path: Path
    env_id: str
    run_id: str
    variant: str
    seed: int
    global_env_step: int | None
    capture_steps: tuple[int, ...]
    frames: tuple[np.ndarray, ...]
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lunarlander-checkpoint", type=Path, default=DEFAULT_LUNARLANDER_CHECKPOINT)
    parser.add_argument("--freeway-checkpoint", type=Path, default=DEFAULT_FREEWAY_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demo-seed-offset", type=int, default=DEFAULT_DEMO_SEED_OFFSET)
    return parser.parse_args()


def _safe_render(env) -> np.ndarray:
    frame = env.render()
    if frame is None:
        raise RuntimeError("Environment returned no RGB frame.")
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise RuntimeError(f"Expected an RGB frame, got shape {frame.shape}.")
    return frame.astype(np.uint8, copy=False)


def capture_checkpoint_snapshots(
    *,
    checkpoint_path: Path,
    title: str,
    capture_steps: tuple[int, ...],
    max_steps: int,
    demo_seed_offset: int,
) -> SnapshotRun:
    payload = load_checkpoint_payload(checkpoint_path)
    env_id = str(payload["config"]["env"]["env_id"])
    seed = int(payload["seed"])
    run_id = str(payload["run_id"])
    variant = str(payload["variant"])
    global_env_step = payload.get("global_env_step")
    if global_env_step is not None:
        global_env_step = int(global_env_step)

    env = None
    frames: list[np.ndarray] = []
    episode_return = 0.0
    episode_length = 0
    terminated = False
    truncated = False
    try:
        env = make_env(env_id, seed=seed, render_mode="rgb_array")
        observation, _info = env.reset(seed=seed + demo_seed_offset)
        network = build_network_from_checkpoint(
            payload,
            observation_shape=tuple(int(dim) for dim in env.observation_space.shape),
            action_dim=int(env.action_space.n),
            device=torch.device("cpu"),
        )

        remaining_steps = set(capture_steps)
        for step in range(1, max_steps + 1):
            action = select_greedy_action(network, observation, torch.device("cpu"))
            observation, reward, terminated, truncated, _info = env.step(action)
            episode_return += float(reward)
            episode_length = step

            if step in remaining_steps:
                frames.append(_safe_render(env))
                remaining_steps.remove(step)
                if not remaining_steps:
                    break

            if terminated or truncated:
                if remaining_steps:
                    frames.append(_safe_render(env))
                break
    finally:
        if env is not None:
            env.close()

    if not frames:
        raise RuntimeError(f"No frames were captured from {checkpoint_path}.")

    while len(frames) < len(capture_steps):
        frames.append(frames[-1].copy())

    return SnapshotRun(
        title=title,
        checkpoint_path=checkpoint_path,
        env_id=env_id,
        run_id=run_id,
        variant=variant,
        seed=seed,
        global_env_step=global_env_step,
        capture_steps=capture_steps,
        frames=tuple(frames[: len(capture_steps)]),
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def draw_panel(fig: plt.Figure, spec, run: SnapshotRun) -> None:
    inner = spec.subgridspec(3, 2, height_ratios=[0.16, 1.0, 1.0], hspace=0.08, wspace=0.04)

    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.45, run.title, ha="center", va="center", fontsize=12, fontweight="bold")

    for index, frame in enumerate(run.frames):
        row = 1 + index // 2
        col = index % 2
        ax = fig.add_subplot(inner[row, col])
        ax.imshow(frame)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_edgecolor("#d0d0d0")


def save_snapshot_figure(runs: tuple[SnapshotRun, SnapshotRun], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 5.6), dpi=300, facecolor="white")
    outer = fig.add_gridspec(1, 2, left=0.02, right=0.98, bottom=0.03, top=0.96, wspace=0.08)
    for index, run in enumerate(runs):
        draw_panel(fig, outer[index], run)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lunarlander = capture_checkpoint_snapshots(
        checkpoint_path=args.lunarlander_checkpoint,
        title="LunarLander-v3 (Double + Dueling DQN)",
        capture_steps=(1, 100, 200, 320),
        max_steps=400,
        demo_seed_offset=args.demo_seed_offset,
    )
    freeway = capture_checkpoint_snapshots(
        checkpoint_path=args.freeway_checkpoint,
        title="Atari Freeway (Double DQN)",
        capture_steps=(1, 500, 1000, 1500),
        max_steps=1600,
        demo_seed_offset=args.demo_seed_offset,
    )
    save_snapshot_figure((lunarlander, freeway), args.output)

    print(f"Wrote {args.output}")
    for run in (lunarlander, freeway):
        step_text = ", ".join(str(step) for step in run.capture_steps)
        status = "terminated" if run.terminated else "truncated" if run.truncated else "active"
        print(
            f"{run.env_id}: {run.run_id}, seed={run.seed}, variant={run.variant}, "
            f"checkpoint_step={run.global_env_step}, capture_steps=[{step_text}], "
            f"rollout_step={run.episode_length}, partial_return={run.episode_return:.2f}, status={status}"
        )


if __name__ == "__main__":
    main()
