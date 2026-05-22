#!/usr/bin/env python3
"""Generate task-selection assets for the report."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rl-analysis-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_LUNAR_ROOT = REPO_ROOT / "output/full_lunarlander_10seeds_20260516_042845"
DEFAULT_FREEWAY_ROOT = REPO_ROOT / "output/run_freeway_upbiased_20260520_110511"
DEFAULT_LUNAR_CHECKPOINT = REPO_ROOT / "selected_checkpoints/lunarlander_agent.pt"
DEFAULT_FREEWAY_CHECKPOINT = REPO_ROOT / "selected_checkpoints/freeway_agent.pt"
DEFAULT_TABLE_CSV = REPO_ROOT / "report/helper/tables/task_selection_overview.csv"
DEFAULT_TABLE_TEX = REPO_ROOT / "report/helper/tables/task_selection_overview.tex"
DEFAULT_FIGURE = REPO_ROOT / "report/helper/figures/task_selection_snapshots.png"


@dataclass(frozen=True)
class TaskSpec:
    task: str
    result_root: Path
    checkpoint: Path
    demo_agent: str
    observation_summary: str
    action_summary: str
    success_signal: str
    analysis_role: str


TASKS = (
    TaskSpec(
        task="Lunar Lander",
        result_root=DEFAULT_LUNAR_ROOT,
        checkpoint=DEFAULT_LUNAR_CHECKPOINT,
        demo_agent="Double + Dueling DQN",
        observation_summary="Vector state with position, velocity, angle, and leg contacts.",
        action_summary="Discrete(4): no-op, left engine, main engine, right engine.",
        success_signal="Episode return at least 200.",
        analysis_role="Shaped rewards and a compact state make it useful for studying stability and sample efficiency.",
    ),
    TaskSpec(
        task="Atari Freeway",
        result_root=DEFAULT_FREEWAY_ROOT,
        checkpoint=DEFAULT_FREEWAY_CHECKPOINT,
        demo_agent="Double DQN",
        observation_summary="Stacked 84 by 84 grayscale Atari frames after preprocessing.",
        action_summary="Reduced Discrete(3): no-op, up, down.",
        success_signal="Score at least 15; one point is earned for each crossing.",
        analysis_role="Sparse visual rewards make it useful for testing representation learning and policy retention.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lunar-root", type=Path, default=DEFAULT_LUNAR_ROOT)
    parser.add_argument("--freeway-root", type=Path, default=DEFAULT_FREEWAY_ROOT)
    parser.add_argument("--lunar-checkpoint", type=Path, default=DEFAULT_LUNAR_CHECKPOINT)
    parser.add_argument("--freeway-checkpoint", type=Path, default=DEFAULT_FREEWAY_CHECKPOINT)
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--table-tex", type=Path, default=DEFAULT_TABLE_TEX)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--refresh-snapshots",
        action="store_true",
        help="Render the snapshot figure from the selected checkpoints instead of only validating it.",
    )
    parser.add_argument("--demo-seed-offset", type=int, default=200_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_run_configs(root: Path) -> list[Path]:
    return sorted(root.rglob("run_config.json"))


def discover_aggregate_summaries(root: Path) -> list[Path]:
    return sorted(root.rglob("aggregate_summary.json"))


def seed_count(paths: list[Path]) -> int:
    seeds: set[int] = set()
    for path in paths:
        match = re.search(r"seed_(\d+)", str(path))
        if match:
            seeds.add(int(match.group(1)))
    return len(seeds)


def variant_count(paths: list[Path]) -> int:
    variants: set[str] = set()
    for path in paths:
        try:
            variants.add(str(read_json(path).get("variant", "")))
        except OSError:
            continue
    return len({variant for variant in variants if variant})


def choose_config(paths: list[Path], preferred_variant: str = "dqn") -> dict[str, Any]:
    for path in paths:
        payload = read_json(path)
        if payload.get("variant") == preferred_variant:
            return payload
    if not paths:
        return {}
    return read_json(paths[0])


def choose_aggregate(paths: list[Path], preferred_variant: str = "dqn") -> dict[str, Any]:
    for path in paths:
        payload = read_json(path)
        if payload.get("variant") == preferred_variant:
            return payload
    if not paths:
        return {}
    return read_json(paths[0])


def threshold_from(config: dict[str, Any], aggregate: dict[str, Any]) -> float | None:
    environment = config.get("environment", {})
    if "success_threshold" in environment:
        return float(environment["success_threshold"])
    steps = aggregate.get("steps_to_threshold", {})
    if "threshold" in steps:
        return float(steps["threshold"])
    return None


def total_steps_from(config: dict[str, Any]) -> int | None:
    budget = config.get("training_budget", {})
    if "total_env_steps" in budget and budget["total_env_steps"] is not None:
        return int(budget["total_env_steps"])
    return None


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def build_overview(args: argparse.Namespace) -> pd.DataFrame:
    task_specs = (
        TASKS[0].__class__(
            **{**TASKS[0].__dict__, "result_root": args.lunar_root, "checkpoint": args.lunar_checkpoint}
        ),
        TASKS[1].__class__(
            **{**TASKS[1].__dict__, "result_root": args.freeway_root, "checkpoint": args.freeway_checkpoint}
        ),
    )

    rows: list[dict[str, Any]] = []
    for spec in task_specs:
        run_configs = discover_run_configs(spec.result_root)
        aggregates = discover_aggregate_summaries(spec.result_root)
        config = choose_config(run_configs)
        aggregate = choose_aggregate(aggregates)
        environment = config.get("environment", {})
        env_id = str(config.get("env_id") or environment.get("env_id") or aggregate.get("env_id"))
        threshold = threshold_from(config, aggregate)
        steps = total_steps_from(config)
        rows.append(
            {
                "task": spec.task,
                "environment_id": env_id,
                "result_root": rel(spec.result_root),
                "observation": spec.observation_summary,
                "action_space": spec.action_summary,
                "success_signal": spec.success_signal,
                "success_threshold": threshold,
                "seed_count": seed_count(run_configs),
                "variant_count": variant_count(run_configs),
                "total_env_steps": steps,
                "network_architecture": environment.get("network_architecture"),
                "selected_demo_agent": spec.demo_agent,
                "selected_checkpoint": rel(spec.checkpoint),
                "analysis_role": spec.analysis_role,
                "snapshot_figure": rel(args.figure),
            }
        )
    return pd.DataFrame(rows)


def latex_escape(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_table_tex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = df[["task", "environment_id", "observation", "action_space", "success_signal", "analysis_role"]]
    headers = ["Task", "Environment", "Observation", "Action Space", "Success Signal", "Reason for Selection"]
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.11\textwidth}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.18\textwidth}>{\raggedright\arraybackslash}p{0.14\textwidth}Y@{}}",
        r"\toprule",
        " & ".join(r"\textbf{" + header + "}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in display.itertuples(index=False):
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_snapshot(path: Path) -> tuple[int, int, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. Re-run with --refresh-snapshots after installing the rendering dependencies."
        )
    image = mpimg.imread(path)
    pixels = np.asarray(image)
    if pixels.ndim < 2:
        raise ValueError(f"Expected an image array, got shape {pixels.shape}.")
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    mean_intensity = float(np.mean(pixels))
    if width < 1000 or height < 500:
        raise ValueError(f"Snapshot image is unexpectedly small: {width}x{height}.")
    if not np.isfinite(mean_intensity):
        raise ValueError("Snapshot image has invalid pixel values.")
    return width, height, mean_intensity


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


def refresh_snapshot_figure(
    *,
    lunar_checkpoint: Path,
    freeway_checkpoint: Path,
    output_path: Path,
    demo_seed_offset: int,
) -> None:
    import torch

    from rl_analysis.demo import build_network_from_checkpoint, load_checkpoint_payload, select_greedy_action
    from rl_analysis.envs import make_env

    def safe_render(env: Any) -> np.ndarray:
        frame = env.render()
        if frame is None:
            raise RuntimeError("Environment returned no RGB frame.")
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError(f"Expected an RGB frame, got shape {frame.shape}.")
        return frame.astype(np.uint8, copy=False)

    def capture(
        *, checkpoint_path: Path, title: str, capture_steps: tuple[int, ...], max_steps: int
    ) -> SnapshotRun:
        payload = load_checkpoint_payload(checkpoint_path)
        env_id = str(payload["config"]["env"]["env_id"])
        seed = int(payload["seed"])
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
                    frames.append(safe_render(env))
                    remaining_steps.remove(step)
                    if not remaining_steps:
                        break
                if terminated or truncated:
                    if remaining_steps:
                        frames.append(safe_render(env))
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
            run_id=str(payload["run_id"]),
            variant=str(payload["variant"]),
            seed=seed,
            global_env_step=int(payload["global_env_step"]) if payload.get("global_env_step") is not None else None,
            capture_steps=capture_steps,
            frames=tuple(frames[: len(capture_steps)]),
            episode_return=episode_return,
            episode_length=episode_length,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )

    def draw_panel(fig: plt.Figure, spec: Any, run: SnapshotRun) -> None:
        inner = spec.subgridspec(3, 2, height_ratios=[0.16, 1.0, 1.0], hspace=0.08, wspace=0.04)
        title_ax = fig.add_subplot(inner[0, :])
        title_ax.axis("off")
        title_ax.text(0.5, 0.45, run.title, ha="center", va="center", fontsize=12, fontweight="bold")
        for index, frame in enumerate(run.frames):
            ax = fig.add_subplot(inner[1 + index // 2, index % 2])
            ax.imshow(frame)
            ax.set_axis_off()
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_edgecolor("#d0d0d0")

    lunar = capture(
        checkpoint_path=lunar_checkpoint,
        title="LunarLander-v3 (Double + Dueling DQN)",
        capture_steps=(1, 100, 200, 320),
        max_steps=400,
    )
    freeway = capture(
        checkpoint_path=freeway_checkpoint,
        title="Atari Freeway (Double DQN)",
        capture_steps=(1, 500, 1000, 1500),
        max_steps=1600,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12.8, 5.6), dpi=300, facecolor="white")
    outer = fig.add_gridspec(1, 2, left=0.02, right=0.98, bottom=0.03, top=0.96, wspace=0.08)
    draw_panel(fig, outer[0], lunar)
    draw_panel(fig, outer[1], freeway)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.refresh_snapshots:
        refresh_snapshot_figure(
            lunar_checkpoint=args.lunar_checkpoint,
            freeway_checkpoint=args.freeway_checkpoint,
            output_path=args.figure,
            demo_seed_offset=args.demo_seed_offset,
        )

    overview = build_overview(args)
    args.table_csv.parent.mkdir(parents=True, exist_ok=True)
    overview.to_csv(args.table_csv, index=False)
    write_table_tex(overview, args.table_tex)
    width, height, mean_intensity = validate_snapshot(args.figure)

    print(f"Wrote {rel(args.table_csv)}")
    print(f"Wrote {rel(args.table_tex)}")
    print(f"Validated {rel(args.figure)} ({width}x{height}, mean pixel={mean_intensity:.3f})")


if __name__ == "__main__":
    main()
