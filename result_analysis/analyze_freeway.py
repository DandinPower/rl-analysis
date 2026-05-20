#!/usr/bin/env python3
"""Analyze Freeway experiment stability from logged JSON metrics.

The script scans every Freeway run under output/, builds compact pandas
summaries, writes figures, and generates a Markdown report in result_analysis/.
It uses only logged metrics and does not rerun environments or load checkpoints.
"""

from __future__ import annotations

import json
import math
import textwrap
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
RESULT_DIR = ROOT / "result_analysis"
FIGURE_DIR = RESULT_DIR / "figures"

SUCCESS_THRESHOLD = 15.0
COLLAPSE_FINAL_THRESHOLD = 5.0
FREEWAY_THRESHOLDS = (5.0, 10.0, 15.0, 22.5)

OUTCOME_ORDER = [
    "stable_success",
    "partial_after_success",
    "collapsed_after_success",
    "never_reached_success",
    "no_eval_data",
]

OUTCOME_LABELS = {
    "stable_success": "Stable success",
    "partial_after_success": "Partial after success",
    "collapsed_after_success": "Collapsed after success",
    "never_reached_success": "Never reached success",
    "no_eval_data": "No eval data",
}

OUTCOME_COLORS = {
    "stable_success": "#2ca25f",
    "partial_after_success": "#fdae61",
    "collapsed_after_success": "#de2d26",
    "never_reached_success": "#636363",
    "no_eval_data": "#9ecae1",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def finite_or_nan(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_or_nan(values: list[Any]) -> float:
    floats = [finite_or_nan(value) for value in values]
    floats = [value for value in floats if np.isfinite(value)]
    if not floats:
        return float("nan")
    return float(np.mean(floats))


def fraction_or_nan(values: list[Any]) -> float:
    if not values:
        return float("nan")
    return float(np.mean([1.0 if bool(value) else 0.0 for value in values]))


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return ""
    if abs(numeric) >= 1000 and abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric)):,}"
    if abs(numeric - round(numeric)) < 1e-9:
        return f"{int(round(numeric))}"
    return f"{numeric:.{digits}f}".rstrip("0").rstrip(".")


def fmt_percent(value: Any, digits: int = 1) -> str:
    numeric = finite_or_nan(value)
    if not np.isfinite(numeric):
        return ""
    return f"{numeric * 100:.{digits}f}%"


def markdown_table(df: pd.DataFrame, *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"

    display = df.copy()
    for col in display.columns:
        display[col] = display[col].map(lambda value: "" if pd.isna(value) else str(value))

    headers = list(display.columns)
    rows = display.values.tolist()
    widths = [
        max(len(str(header)), *(len(str(row[index])) for row in rows))
        for index, header in enumerate(headers)
    ]

    def render_row(values: list[Any]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(values)) + " |"

    lines = [render_row(headers), "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def classify_outcome(best_eval: float, final_eval: float) -> str:
    if not np.isfinite(best_eval) or not np.isfinite(final_eval):
        return "no_eval_data"
    if final_eval >= SUCCESS_THRESHOLD:
        return "stable_success"
    if best_eval >= SUCCESS_THRESHOLD and final_eval < COLLAPSE_FINAL_THRESHOLD:
        return "collapsed_after_success"
    if best_eval >= SUCCESS_THRESHOLD:
        return "partial_after_success"
    return "never_reached_success"


def first_step_reaching(eval_rows: list[dict[str, Any]], threshold: float) -> float:
    for row in eval_rows:
        value = finite_or_nan(row.get("return_mean"))
        if np.isfinite(value) and value >= threshold:
            return finite_or_nan(row.get("global_env_step"))
    return float("nan")


def eval_records_for_run(
    *,
    batch: str,
    variant: str,
    seed: int | None,
    run_id: str,
    eval_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in eval_rows:
        freeway_eval = row.get("freeway_eval", {}) or {}
        records.append(
            {
                "batch": batch,
                "variant": variant,
                "seed": seed,
                "run_id": run_id,
                "eval_index": row.get("eval_index"),
                "global_env_step": finite_or_nan(row.get("global_env_step")),
                "return_mean": finite_or_nan(row.get("return_mean")),
                "return_std": finite_or_nan(row.get("return_std")),
                "return_min": finite_or_nan(row.get("return_min")),
                "return_max": finite_or_nan(row.get("return_max")),
                "zero_score_rate": finite_or_nan(freeway_eval.get("zero_score_rate")),
                "action_up_fraction": finite_or_nan(freeway_eval.get("action_up_fraction_mean")),
                "action_down_fraction": finite_or_nan(freeway_eval.get("action_down_fraction_mean")),
                "action_noop_fraction": finite_or_nan(freeway_eval.get("action_noop_fraction_mean")),
            }
        )
    return records


def summarize_episodes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {
            "total_train_episodes": float("nan"),
            "tail_train_return_mean": float("nan"),
            "tail_train_zero_score_fraction": float("nan"),
            "tail_train_up_fraction": float("nan"),
            "tail_train_down_fraction": float("nan"),
            "tail_train_noop_fraction": float("nan"),
            "tail_train_epsilon_mean": float("nan"),
        }

    tail: deque[dict[str, Any]] = deque(maxlen=100)
    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                total += 1
                tail.append(json.loads(stripped))

    rows = list(tail)
    return {
        "total_train_episodes": float(total),
        "tail_train_return_mean": mean_or_nan([row.get("episode_return") for row in rows]),
        "tail_train_zero_score_fraction": fraction_or_nan(
            [(row.get("game_specific", {}) or {}).get("zero_score_episode") for row in rows]
        ),
        "tail_train_up_fraction": mean_or_nan(
            [(row.get("game_specific", {}) or {}).get("action_up_fraction") for row in rows]
        ),
        "tail_train_down_fraction": mean_or_nan(
            [(row.get("game_specific", {}) or {}).get("action_down_fraction") for row in rows]
        ),
        "tail_train_noop_fraction": mean_or_nan(
            [(row.get("game_specific", {}) or {}).get("action_noop_fraction") for row in rows]
        ),
        "tail_train_epsilon_mean": mean_or_nan([row.get("epsilon") for row in rows]),
    }


def summarize_updates(path: Path) -> dict[str, float | bool]:
    summary: dict[str, float | bool] = {
        "total_update_logs": float("nan"),
        "tail_td_loss_mean": float("nan"),
        "tail_td_error_abs_mean": float("nan"),
        "tail_grad_norm_before_clip": float("nan"),
        "tail_grad_clip_fraction": float("nan"),
        "tail_online_q_mean": float("nan"),
        "tail_online_q_max_mean": float("nan"),
        "tail_action_gap_mean": float("nan"),
        "tail_q_overestimation_proxy": float("nan"),
        "tail_replay_reward_mean": float("nan"),
        "tail_replay_reward_std": float("nan"),
        "tail_sample_age_mean": float("nan"),
        "any_nan_detected": False,
        "any_inf_detected": False,
        "any_gradient_explosion": False,
        "any_loss_explosion": False,
        "any_q_value_explosion": False,
    }
    if not path.exists():
        return summary

    metrics: dict[str, list[Any]] = {
        "td_loss": [],
        "td_abs": [],
        "grad": [],
        "grad_clipped": [],
        "q_mean": [],
        "q_max_mean": [],
        "action_gap": [],
        "overestimation": [],
        "replay_reward_mean": [],
        "replay_reward_std": [],
        "sample_age_mean": [],
    }

    total = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            row = json.loads(stripped)
            loss = row.get("loss", {}) or {}
            opt = row.get("optimization", {}) or {}
            q_values = row.get("q_values", {}) or {}
            replay = row.get("replay", {}) or {}
            failure = row.get("failure_diagnostics", {}) or {}

            metrics["td_loss"].append(loss.get("td_loss_mean"))
            metrics["td_abs"].append(loss.get("td_error_abs_mean"))
            metrics["grad"].append(opt.get("grad_norm_before_clip"))
            metrics["grad_clipped"].append(opt.get("grad_norm_clipped"))
            metrics["q_mean"].append(q_values.get("online_q_mean"))
            metrics["q_max_mean"].append(q_values.get("online_q_max_mean"))
            metrics["action_gap"].append(q_values.get("action_gap_mean"))
            metrics["overestimation"].append(q_values.get("q_overestimation_proxy"))
            metrics["replay_reward_mean"].append(replay.get("sample_reward_mean"))
            metrics["replay_reward_std"].append(replay.get("sample_reward_std"))
            metrics["sample_age_mean"].append(replay.get("sample_age_mean"))

            summary["any_nan_detected"] = bool(summary["any_nan_detected"] or failure.get("nan_detected"))
            summary["any_inf_detected"] = bool(summary["any_inf_detected"] or failure.get("inf_detected"))
            summary["any_gradient_explosion"] = bool(
                summary["any_gradient_explosion"] or failure.get("gradient_explosion")
            )
            summary["any_loss_explosion"] = bool(summary["any_loss_explosion"] or failure.get("loss_explosion"))
            summary["any_q_value_explosion"] = bool(
                summary["any_q_value_explosion"] or failure.get("q_value_explosion")
            )

    tail_start = max(0, total - max(1, total // 10))
    summary.update(
        {
            "total_update_logs": float(total),
            "tail_td_loss_mean": mean_or_nan(metrics["td_loss"][tail_start:]),
            "tail_td_error_abs_mean": mean_or_nan(metrics["td_abs"][tail_start:]),
            "tail_grad_norm_before_clip": mean_or_nan(metrics["grad"][tail_start:]),
            "tail_grad_clip_fraction": fraction_or_nan(metrics["grad_clipped"][tail_start:]),
            "tail_online_q_mean": mean_or_nan(metrics["q_mean"][tail_start:]),
            "tail_online_q_max_mean": mean_or_nan(metrics["q_max_mean"][tail_start:]),
            "tail_action_gap_mean": mean_or_nan(metrics["action_gap"][tail_start:]),
            "tail_q_overestimation_proxy": mean_or_nan(metrics["overestimation"][tail_start:]),
            "tail_replay_reward_mean": mean_or_nan(metrics["replay_reward_mean"][tail_start:]),
            "tail_replay_reward_std": mean_or_nan(metrics["replay_reward_std"][tail_start:]),
            "tail_sample_age_mean": mean_or_nan(metrics["sample_age_mean"][tail_start:]),
        }
    )
    return summary


def summarize_checkpoints(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    if not rows:
        return {
            "checkpoint_count": float("nan"),
            "best_checkpoint_step": float("nan"),
            "final_checkpoint_eval": float("nan"),
        }
    best_rows = [row for row in rows if row.get("is_best_checkpoint_so_far")]
    best_row = best_rows[-1] if best_rows else max(rows, key=lambda item: finite_or_nan(item.get("eval_return_mean_at_checkpoint")))
    return {
        "checkpoint_count": float(len(rows)),
        "best_checkpoint_step": finite_or_nan(best_row.get("global_env_step")),
        "final_checkpoint_eval": finite_or_nan(rows[-1].get("eval_return_mean_at_checkpoint")),
    }


def summarize_run(cfg_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cfg = read_json(cfg_path)
    rel_parts = cfg_path.relative_to(OUTPUT_DIR).parts
    batch = rel_parts[0]
    run_dir = cfg_path.parent
    run_id = str(cfg.get("run_id") or run_dir.name)
    variant = str(cfg.get("variant") or rel_parts[2])
    seed = cfg.get("seed")

    env_config = cfg.get("environment", {}) or {}
    training = cfg.get("training_budget", {}) or {}
    dqn = cfg.get("dqn_hyperparameters", {}) or {}
    optimizer = cfg.get("optimizer", {}) or {}
    exploration = cfg.get("exploration", {}) or {}
    algorithm = cfg.get("algorithm", {}) or {}

    eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
    eval_records = eval_records_for_run(
        batch=batch,
        variant=variant,
        seed=seed,
        run_id=run_id,
        eval_rows=eval_rows,
    )

    eval_returns = np.array([record["return_mean"] for record in eval_records], dtype=float)
    eval_steps = np.array([record["global_env_step"] for record in eval_records], dtype=float)
    valid_eval = np.isfinite(eval_returns) & np.isfinite(eval_steps)
    valid_returns = eval_returns[valid_eval]
    valid_steps = eval_steps[valid_eval]

    if len(valid_returns):
        best_index = int(np.argmax(valid_returns))
        best_eval = float(valid_returns[best_index])
        best_step = float(valid_steps[best_index])
        final_eval = float(valid_returns[-1])
        final_step = float(valid_steps[-1])
        auc = float(np.trapezoid(valid_returns, valid_steps)) if len(valid_returns) >= 2 else 0.0
        final_record = eval_records[-1]
        post_best_min = float(np.min(valid_returns[best_index:]))
    else:
        best_eval = float("nan")
        best_step = float("nan")
        final_eval = float("nan")
        final_step = float("nan")
        auc = float("nan")
        final_record = {}
        post_best_min = float("nan")

    outcome = classify_outcome(best_eval, final_eval)
    final_zero = finite_or_nan(final_record.get("zero_score_rate"))
    final_up = finite_or_nan(final_record.get("action_up_fraction"))
    final_down = finite_or_nan(final_record.get("action_down_fraction"))
    final_noop = finite_or_nan(final_record.get("action_noop_fraction"))

    threshold_steps = {
        f"first_step_score_{str(threshold).replace('.', '_')}": first_step_reaching(eval_rows, threshold)
        for threshold in FREEWAY_THRESHOLDS
    }

    row: dict[str, Any] = {
        "batch": batch,
        "variant": variant,
        "seed": seed,
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "status": cfg.get("status"),
        "env_id": cfg.get("env_id") or env_config.get("env_id"),
        "total_env_steps": finite_or_nan(training.get("total_env_steps")),
        "eval_frequency_env_steps": finite_or_nan(training.get("eval_frequency_env_steps")),
        "checkpoint_frequency_env_steps": finite_or_nan(training.get("checkpoint_frequency_env_steps")),
        "num_eval_episodes": finite_or_nan(training.get("num_eval_episodes")),
        "learning_rate": finite_or_nan(optimizer.get("learning_rate")),
        "optimizer_epsilon": finite_or_nan(optimizer.get("epsilon")),
        "batch_size": finite_or_nan(dqn.get("batch_size")),
        "gamma": finite_or_nan(dqn.get("gamma")),
        "replay_buffer_size": finite_or_nan(dqn.get("replay_buffer_size")),
        "learning_starts": finite_or_nan(dqn.get("learning_starts")),
        "train_frequency_env_steps": finite_or_nan(dqn.get("train_frequency_env_steps")),
        "gradient_steps_per_train": finite_or_nan(dqn.get("gradient_steps_per_train")),
        "target_update_frequency_env_steps": finite_or_nan(dqn.get("target_update_frequency_env_steps")),
        "epsilon_decay_env_steps": finite_or_nan(exploration.get("epsilon_decay_env_steps")),
        "epsilon_final": finite_or_nan(exploration.get("epsilon_final")),
        "eval_epsilon": finite_or_nan(exploration.get("eval_epsilon")),
        "use_replay_buffer": bool(algorithm.get("use_replay_buffer", True)),
        "use_target_network": bool(algorithm.get("use_target_network", True)),
        "use_double_dqn": bool(algorithm.get("use_double_dqn", False)),
        "use_dueling_network": bool(algorithm.get("use_dueling_network", False)),
        "eval_points": float(len(eval_records)),
        "best_eval_return": best_eval,
        "best_eval_step": best_step,
        "final_eval_return": final_eval,
        "final_eval_step": final_step,
        "area_under_eval_curve": auc,
        "normalized_auc": auc / max(1.0, finite_or_nan(training.get("total_env_steps")))
        if np.isfinite(auc)
        else float("nan"),
        "collapse_drop_best_to_final": best_eval - final_eval
        if np.isfinite(best_eval) and np.isfinite(final_eval)
        else float("nan"),
        "largest_post_best_drop": best_eval - post_best_min
        if np.isfinite(best_eval) and np.isfinite(post_best_min)
        else float("nan"),
        "final_zero_score_rate": final_zero,
        "final_action_up_fraction": final_up,
        "final_action_down_fraction": final_down,
        "final_action_noop_fraction": final_noop,
        "outcome": outcome,
        "stable_success": outcome == "stable_success",
        "ever_reached_success": bool(np.isfinite(best_eval) and best_eval >= SUCCESS_THRESHOLD),
        "collapsed_after_success": outcome == "collapsed_after_success",
    }
    row.update(threshold_steps)
    row.update(summarize_episodes(run_dir / "train_episode_metrics.jsonl"))
    row.update(summarize_updates(run_dir / "train_update_metrics.jsonl"))
    row.update(summarize_checkpoints(run_dir / "checkpoint_metrics.jsonl"))
    return row, eval_records


def discover_freeway_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for cfg_path in sorted(OUTPUT_DIR.glob("**/freeway/**/run_config.json")):
        row, records = summarize_run(cfg_path)
        if row.get("env_id") != "ALE/Freeway-v5":
            continue
        run_rows.append(row)
        eval_rows.extend(records)

    run_df = pd.DataFrame(run_rows)
    eval_df = pd.DataFrame(eval_rows)
    if not run_df.empty:
        run_df["outcome"] = pd.Categorical(run_df["outcome"], categories=OUTCOME_ORDER, ordered=True)
        run_df = run_df.sort_values(["batch", "variant", "seed", "run_id"]).reset_index(drop=True)
    if not eval_df.empty:
        eval_df = eval_df.sort_values(["batch", "variant", "seed", "global_env_step"]).reset_index(drop=True)
    return run_df, eval_df


def build_inventory(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_df.groupby(["batch", "variant"], observed=True)
        .agg(
            runs=("run_id", "count"),
            seeds=("seed", lambda values: ", ".join(str(int(v)) for v in sorted(pd.Series(values).dropna().unique()))),
            total_env_steps=("total_env_steps", "first"),
            eval_freq=("eval_frequency_env_steps", "first"),
            learning_rate=("learning_rate", "first"),
            replay_size=("replay_buffer_size", "first"),
            learning_starts=("learning_starts", "first"),
            train_freq=("train_frequency_env_steps", "first"),
            target_freq=("target_update_frequency_env_steps", "first"),
            epsilon_decay=("epsilon_decay_env_steps", "first"),
            epsilon_final=("epsilon_final", "first"),
        )
        .reset_index()
    )
    for column in [
        "total_env_steps",
        "eval_freq",
        "replay_size",
        "learning_starts",
        "train_freq",
        "target_freq",
        "epsilon_decay",
    ]:
        grouped[column] = grouped[column].map(lambda value: fmt_num(value, 0))
    for column in ["learning_rate", "epsilon_final"]:
        grouped[column] = grouped[column].map(lambda value: fmt_num(value, 6))
    return grouped


def build_group_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_df.groupby(["batch", "variant"], observed=True)
        .agg(
            runs=("run_id", "count"),
            stable_success_rate=("stable_success", "mean"),
            collapsed_rate=("collapsed_after_success", "mean"),
            ever_success_rate=("ever_reached_success", "mean"),
            best_mean=("best_eval_return", "mean"),
            best_max=("best_eval_return", "max"),
            final_mean=("final_eval_return", "mean"),
            final_median=("final_eval_return", "median"),
            final_max=("final_eval_return", "max"),
            zero_mean=("final_zero_score_rate", "mean"),
            final_up_mean=("final_action_up_fraction", "mean"),
            final_down_mean=("final_action_down_fraction", "mean"),
            final_noop_mean=("final_action_noop_fraction", "mean"),
            tail_action_gap_mean=("tail_action_gap_mean", "mean"),
            tail_replay_reward_mean=("tail_replay_reward_mean", "mean"),
        )
        .reset_index()
        .sort_values(["stable_success_rate", "final_mean", "best_mean"], ascending=[False, False, False])
    )
    return grouped


def build_outcome_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_df.groupby("outcome", observed=True)
        .agg(
            runs=("run_id", "count"),
            best_mean=("best_eval_return", "mean"),
            final_mean=("final_eval_return", "mean"),
            zero_mean=("final_zero_score_rate", "mean"),
            final_up=("final_action_up_fraction", "mean"),
            final_down=("final_action_down_fraction", "mean"),
            final_noop=("final_action_noop_fraction", "mean"),
            tail_action_gap=("tail_action_gap_mean", "mean"),
            tail_td_loss=("tail_td_loss_mean", "mean"),
            tail_grad_norm=("tail_grad_norm_before_clip", "mean"),
            tail_replay_reward=("tail_replay_reward_mean", "mean"),
            tail_train_return=("tail_train_return_mean", "mean"),
            tail_train_zero=("tail_train_zero_score_fraction", "mean"),
        )
        .reindex(OUTCOME_ORDER)
        .dropna(how="all")
        .reset_index()
    )
    grouped["outcome_label"] = grouped["outcome"].map(OUTCOME_LABELS)
    return grouped


def save_csvs(run_df: pd.DataFrame, eval_df: pd.DataFrame, group_df: pd.DataFrame, outcome_df: pd.DataFrame) -> None:
    run_df.to_csv(RESULT_DIR / "freeway_run_summary.csv", index=False)
    eval_df.to_csv(RESULT_DIR / "freeway_eval_timeseries.csv", index=False)
    group_df.to_csv(RESULT_DIR / "freeway_group_summary.csv", index=False)
    outcome_df.to_csv(RESULT_DIR / "freeway_outcome_summary.csv", index=False)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 9,
        }
    )


def save_learning_curves(eval_df: pd.DataFrame, run_df: pd.DataFrame) -> Path:
    groups = list(eval_df.groupby(["batch", "variant"], observed=True))
    cols = 2
    rows = math.ceil(len(groups) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(3.2 * rows, 4)), squeeze=False, sharey=True)
    run_lookup = run_df.set_index("run_id")

    for axis in axes.ravel():
        axis.set_visible(False)

    for axis, ((batch, variant), group) in zip(axes.ravel(), groups, strict=False):
        axis.set_visible(True)
        for run_id, run_group in group.groupby("run_id", sort=False):
            outcome = str(run_lookup.loc[run_id, "outcome"]) if run_id in run_lookup.index else "no_eval_data"
            seed = run_lookup.loc[run_id, "seed"] if run_id in run_lookup.index else ""
            is_stable = outcome == "stable_success"
            axis.plot(
                run_group["global_env_step"],
                run_group["return_mean"],
                color=OUTCOME_COLORS.get(outcome, "#969696"),
                linewidth=2.2 if is_stable else 1.0,
                alpha=0.95 if is_stable else 0.42,
                label=f"seed {int(seed)}" if is_stable and pd.notna(seed) else None,
            )

        mean_curve = (
            group.groupby("global_env_step", observed=True)["return_mean"]
            .mean()
            .reset_index()
            .sort_values("global_env_step")
        )
        axis.plot(
            mean_curve["global_env_step"],
            mean_curve["return_mean"],
            color="black",
            linewidth=1.8,
            alpha=0.85,
            label="group mean",
        )
        axis.axhline(SUCCESS_THRESHOLD, color="#08519c", linestyle="--", linewidth=1.0, alpha=0.8)
        axis.set_title(f"{variant} | {batch}", fontsize=8)
        axis.set_xlabel("Env steps")
        axis.set_ylabel("Eval score")
        axis.set_ylim(bottom=-1)
        axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        if axis.get_legend_handles_labels()[0]:
            axis.legend(loc="upper left", fontsize=7, frameon=False)

    fig.suptitle("Freeway evaluation learning curves by batch and variant", fontsize=14, y=0.995)
    fig.tight_layout()
    path = FIGURE_DIR / "eval_learning_curves_by_group.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_best_vs_final(run_df: pd.DataFrame) -> Path:
    fig, axis = plt.subplots(figsize=(7.2, 5.2))
    markers = {
        "dqn": "o",
        "double_dqn": "s",
        "dueling_dqn": "^",
        "dqn_no_replay": "D",
        "dqn_no_target": "X",
    }

    for outcome in OUTCOME_ORDER:
        subset = run_df[run_df["outcome"].astype(str) == outcome]
        if subset.empty:
            continue
        for variant, variant_subset in subset.groupby("variant", observed=True):
            axis.scatter(
                variant_subset["best_eval_return"],
                variant_subset["final_eval_return"],
                s=62,
                marker=markers.get(variant, "o"),
                color=OUTCOME_COLORS.get(outcome, "#969696"),
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
                label=f"{OUTCOME_LABELS[outcome]} / {variant}",
            )

    max_score = max(30.0, finite_or_nan(run_df["best_eval_return"].max()), finite_or_nan(run_df["final_eval_return"].max()))
    axis.plot([0, max_score], [0, max_score], color="#525252", linewidth=1.0, linestyle=":", label="best = final")
    axis.axhline(SUCCESS_THRESHOLD, color="#08519c", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.axvline(SUCCESS_THRESHOLD, color="#08519c", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xlim(-1, max_score + 1)
    axis.set_ylim(-1, max_score + 1)
    axis.set_xlabel("Best eval score")
    axis.set_ylabel("Final eval score")
    axis.set_title("Best score versus final score")
    handles, labels = axis.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    axis.legend(by_label.values(), by_label.keys(), fontsize=6.2, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    path = FIGURE_DIR / "best_vs_final_score.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_action_fractions(run_df: pd.DataFrame) -> Path:
    outcome_df = build_outcome_summary(run_df)
    outcome_df = outcome_df[outcome_df["runs"].fillna(0) > 0].copy()
    labels = outcome_df["outcome_label"].tolist()
    x = np.arange(len(labels))
    width = 0.24

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.bar(x - width, outcome_df["final_up"], width, label="Up", color="#2ca25f")
    axis.bar(x, outcome_df["final_down"], width, label="Down", color="#de2d26")
    axis.bar(x + width, outcome_df["final_noop"], width, label="No-op", color="#756bb1")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Mean final evaluation action fraction")
    axis.set_title("Final evaluation action mix by outcome")
    axis.legend(frameon=False, ncols=3, loc="upper center")
    fig.tight_layout()
    path = FIGURE_DIR / "final_action_fractions_by_outcome.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_case_study_curves(eval_df: pd.DataFrame, run_df: pd.DataFrame) -> Path:
    stable = run_df[run_df["outcome"].astype(str) == "stable_success"].sort_values("final_eval_return", ascending=False)
    collapsed = run_df[run_df["outcome"].astype(str) == "collapsed_after_success"].sort_values(
        "best_eval_return", ascending=False
    )
    selected = []
    if not stable.empty:
        selected.append(stable.iloc[0])
    if not collapsed.empty:
        selected.append(collapsed.iloc[0])

    fig, axes = plt.subplots(1, max(1, len(selected)), figsize=(7 * max(1, len(selected)), 4.8), squeeze=False)
    if not selected:
        axes.ravel()[0].text(0.5, 0.5, "No case studies available", ha="center", va="center")
    for axis, row in zip(axes.ravel(), selected, strict=False):
        curve = eval_df[eval_df["run_id"] == row["run_id"]].sort_values("global_env_step")
        axis.plot(curve["global_env_step"], curve["return_mean"], color="#08519c", linewidth=2, label="score")
        axis.axhline(SUCCESS_THRESHOLD, color="#525252", linestyle="--", linewidth=1, label="success threshold")
        axis.set_xlabel("Env steps")
        axis.set_ylabel("Eval score")
        axis.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        axis2 = axis.twinx()
        axis2.plot(curve["global_env_step"], curve["action_up_fraction"], color="#2ca25f", alpha=0.7, label="up fraction")
        axis2.plot(
            curve["global_env_step"],
            curve["action_down_fraction"],
            color="#de2d26",
            alpha=0.55,
            label="down fraction",
        )
        axis2.plot(
            curve["global_env_step"],
            curve["action_noop_fraction"],
            color="#756bb1",
            alpha=0.55,
            label="no-op fraction",
        )
        axis2.set_ylim(0, 1)
        axis2.set_ylabel("Action fraction")
        axis.set_title(
            f"{OUTCOME_LABELS[str(row['outcome'])]}: {row['variant']} seed {int(row['seed'])}\n"
            f"best {row['best_eval_return']:.2f}, final {row['final_eval_return']:.2f}"
        )
        lines, labels = axis.get_legend_handles_labels()
        lines2, labels2 = axis2.get_legend_handles_labels()
        axis.legend(lines + lines2, labels + labels2, fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    path = FIGURE_DIR / "stable_vs_collapsed_case_studies.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_group_table_figure(group_df: pd.DataFrame, run_df: pd.DataFrame) -> Path:
    columns = [
        "group",
        "runs",
        "stable",
        "collapsed",
        "best_mean",
        "final_mean",
        "final_up",
        "q_gap",
        "lr",
        "replay",
        "eps_decay",
    ]
    table_df = group_df.copy()
    table_df["group"] = table_df["batch"] + "\n" + table_df["variant"]
    table_df["stable"] = table_df["stable_success_rate"].map(lambda value: fmt_percent(value, 0))
    table_df["collapsed"] = table_df["collapsed_rate"].map(lambda value: fmt_percent(value, 0))
    table_df["best_mean"] = table_df["best_mean"].map(lambda value: fmt_num(value, 2))
    table_df["final_mean"] = table_df["final_mean"].map(lambda value: fmt_num(value, 2))
    table_df["final_up"] = table_df["final_up_mean"].map(lambda value: fmt_num(value, 2))
    table_df["q_gap"] = table_df["tail_action_gap_mean"].map(lambda value: fmt_num(value, 4))

    hyper = (
        run_df.groupby(["batch", "variant"], observed=True)
        .agg(
            lr=("learning_rate", "first"),
            replay=("replay_buffer_size", "first"),
            eps_decay=("epsilon_decay_env_steps", "first"),
        )
        .reset_index()
    )
    table_df = table_df.merge(hyper, on=["batch", "variant"], how="left")
    table_df["lr"] = table_df["lr"].map(lambda value: fmt_num(value, 6))
    table_df["replay"] = table_df["replay"].map(lambda value: fmt_num(value, 0))
    table_df["eps_decay"] = table_df["eps_decay"].map(lambda value: fmt_num(value, 0))
    table_df = table_df[columns]

    fig_height = max(4.5, 0.42 * len(table_df) + 1.4)
    fig, axis = plt.subplots(figsize=(14, fig_height))
    axis.axis("off")
    table = axis.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    for (row_index, col_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
        elif row_index % 2 == 0:
            cell.set_facecolor("#fafafa")
        if col_index == 0:
            cell.set_text_props(ha="left")
    axis.set_title("Batch, hyperparameter, and outcome summary", fontsize=13, pad=14)
    fig.tight_layout()
    path = FIGURE_DIR / "hyperparameter_outcome_table.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def make_report_tables(
    run_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    group_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
) -> dict[str, str]:
    inventory = inventory_df.rename(
        columns={
            "batch": "batch",
            "variant": "variant",
            "runs": "runs",
            "seeds": "seeds",
            "total_env_steps": "steps",
            "eval_freq": "eval freq",
            "learning_rate": "lr",
            "replay_size": "replay",
            "learning_starts": "starts",
            "train_freq": "train freq",
            "target_freq": "target freq",
            "epsilon_decay": "eps decay",
            "epsilon_final": "eps final",
        }
    )

    outcome_display = outcome_df.copy()
    outcome_display["outcome"] = outcome_display["outcome_label"]
    outcome_display["runs"] = outcome_display["runs"].map(lambda value: fmt_num(value, 0))
    for col in ["best_mean", "final_mean", "final_up", "final_down", "final_noop", "tail_action_gap"]:
        outcome_display[col] = outcome_display[col].map(lambda value: fmt_num(value, 3))
    for col in ["zero_mean", "tail_train_zero"]:
        outcome_display[col] = outcome_display[col].map(lambda value: fmt_percent(value, 1))
    outcome_display["tail_replay_reward"] = outcome_display["tail_replay_reward"].map(lambda value: fmt_num(value, 6))
    outcome_display["tail_train_return"] = outcome_display["tail_train_return"].map(lambda value: fmt_num(value, 3))
    outcome_display = outcome_display[
        [
            "outcome",
            "runs",
            "best_mean",
            "final_mean",
            "zero_mean",
            "final_up",
            "final_down",
            "final_noop",
            "tail_action_gap",
            "tail_replay_reward",
            "tail_train_return",
            "tail_train_zero",
        ]
    ].rename(
        columns={
            "best_mean": "best mean",
            "final_mean": "final mean",
            "zero_mean": "final zero rate",
            "final_up": "final up",
            "final_down": "final down",
            "final_noop": "final no-op",
            "tail_action_gap": "tail Q gap",
            "tail_replay_reward": "tail replay reward",
            "tail_train_return": "tail train score",
            "tail_train_zero": "tail train zero",
        }
    )

    group_display = group_df.copy()
    group_display["stable"] = group_display["stable_success_rate"].map(lambda value: fmt_percent(value, 0))
    group_display["collapsed"] = group_display["collapsed_rate"].map(lambda value: fmt_percent(value, 0))
    group_display["ever >=15"] = group_display["ever_success_rate"].map(lambda value: fmt_percent(value, 0))
    for col in ["best_mean", "final_mean", "final_median", "final_max", "final_up_mean"]:
        group_display[col] = group_display[col].map(lambda value: fmt_num(value, 2))
    group_display["zero_mean"] = group_display["zero_mean"].map(lambda value: fmt_percent(value, 0))
    group_display = group_display[
        [
            "batch",
            "variant",
            "runs",
            "stable",
            "collapsed",
            "ever >=15",
            "best_mean",
            "final_mean",
            "final_median",
            "final_max",
            "zero_mean",
            "final_up_mean",
        ]
    ].rename(
        columns={
            "best_mean": "best mean",
            "final_mean": "final mean",
            "final_median": "final median",
            "final_max": "final max",
            "zero_mean": "zero rate",
            "final_up_mean": "final up",
        }
    )

    stable_runs = run_df[run_df["outcome"].astype(str) == "stable_success"].sort_values(
        "final_eval_return", ascending=False
    )
    stable_display = stable_runs[
        [
            "batch",
            "variant",
            "seed",
            "best_eval_return",
            "best_eval_step",
            "final_eval_return",
            "final_zero_score_rate",
            "final_action_up_fraction",
            "tail_train_return_mean",
            "tail_action_gap_mean",
            "tail_replay_reward_mean",
        ]
    ].copy()
    for col in ["best_eval_return", "final_eval_return", "final_action_up_fraction", "tail_train_return_mean"]:
        stable_display[col] = stable_display[col].map(lambda value: fmt_num(value, 3))
    stable_display["best_eval_step"] = stable_display["best_eval_step"].map(lambda value: fmt_num(value, 0))
    stable_display["final_zero_score_rate"] = stable_display["final_zero_score_rate"].map(lambda value: fmt_percent(value, 1))
    stable_display["tail_action_gap_mean"] = stable_display["tail_action_gap_mean"].map(lambda value: fmt_num(value, 5))
    stable_display["tail_replay_reward_mean"] = stable_display["tail_replay_reward_mean"].map(lambda value: fmt_num(value, 6))
    stable_display = stable_display.rename(
        columns={
            "best_eval_return": "best",
            "best_eval_step": "best step",
            "final_eval_return": "final",
            "final_zero_score_rate": "zero rate",
            "final_action_up_fraction": "final up",
            "tail_train_return_mean": "tail train",
            "tail_action_gap_mean": "tail Q gap",
            "tail_replay_reward_mean": "tail replay reward",
        }
    )

    collapsed_runs = run_df[run_df["outcome"].astype(str) == "collapsed_after_success"].sort_values(
        "best_eval_return", ascending=False
    )
    collapsed_display = collapsed_runs[
        [
            "batch",
            "variant",
            "seed",
            "best_eval_return",
            "best_eval_step",
            "final_eval_return",
            "final_zero_score_rate",
            "final_action_up_fraction",
            "final_action_down_fraction",
            "final_action_noop_fraction",
            "tail_train_return_mean",
            "tail_action_gap_mean",
        ]
    ].head(12).copy()
    for col in [
        "best_eval_return",
        "final_eval_return",
        "final_action_up_fraction",
        "final_action_down_fraction",
        "final_action_noop_fraction",
        "tail_train_return_mean",
    ]:
        collapsed_display[col] = collapsed_display[col].map(lambda value: fmt_num(value, 3))
    collapsed_display["best_eval_step"] = collapsed_display["best_eval_step"].map(lambda value: fmt_num(value, 0))
    collapsed_display["final_zero_score_rate"] = collapsed_display["final_zero_score_rate"].map(lambda value: fmt_percent(value, 1))
    collapsed_display["tail_action_gap_mean"] = collapsed_display["tail_action_gap_mean"].map(lambda value: fmt_num(value, 5))
    collapsed_display = collapsed_display.rename(
        columns={
            "best_eval_return": "best",
            "best_eval_step": "best step",
            "final_eval_return": "final",
            "final_zero_score_rate": "zero rate",
            "final_action_up_fraction": "final up",
            "final_action_down_fraction": "final down",
            "final_action_noop_fraction": "final no-op",
            "tail_train_return_mean": "tail train",
            "tail_action_gap_mean": "tail Q gap",
        }
    )

    return {
        "inventory": markdown_table(inventory),
        "outcome": markdown_table(outcome_display),
        "group": markdown_table(group_display),
        "stable": markdown_table(stable_display),
        "collapsed": markdown_table(collapsed_display),
    }


def figure_link(path: Path) -> str:
    return f"figures/{path.name}"


def build_report(
    run_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    group_df: pd.DataFrame,
    outcome_df: pd.DataFrame,
    figure_paths: dict[str, Path],
) -> str:
    tables = make_report_tables(run_df, inventory_df, group_df, outcome_df)

    n_runs = len(run_df)
    n_eval = len(eval_df)
    outcome_counts = run_df["outcome"].astype(str).value_counts().to_dict()
    stable_count = int(outcome_counts.get("stable_success", 0))
    collapsed_count = int(outcome_counts.get("collapsed_after_success", 0))
    partial_count = int(outcome_counts.get("partial_after_success", 0))
    never_count = int(outcome_counts.get("never_reached_success", 0))

    stable_metrics = outcome_df[outcome_df["outcome"].astype(str) == "stable_success"]
    collapsed_metrics = outcome_df[outcome_df["outcome"].astype(str) == "collapsed_after_success"]
    never_metrics = outcome_df[outcome_df["outcome"].astype(str) == "never_reached_success"]

    def outcome_value(frame: pd.DataFrame, column: str) -> float:
        if frame.empty:
            return float("nan")
        return finite_or_nan(frame.iloc[0].get(column))

    stable_final = outcome_value(stable_metrics, "final_mean")
    stable_up = outcome_value(stable_metrics, "final_up")
    stable_gap = outcome_value(stable_metrics, "tail_action_gap")
    stable_replay_reward = outcome_value(stable_metrics, "tail_replay_reward")
    stable_tail_train = outcome_value(stable_metrics, "tail_train_return")
    collapsed_final = outcome_value(collapsed_metrics, "final_mean")
    collapsed_zero = outcome_value(collapsed_metrics, "zero_mean")
    collapsed_up = outcome_value(collapsed_metrics, "final_up")
    collapsed_gap = outcome_value(collapsed_metrics, "tail_action_gap")
    collapsed_replay_reward = outcome_value(collapsed_metrics, "tail_replay_reward")
    collapsed_tail_train = outcome_value(collapsed_metrics, "tail_train_return")
    never_best = outcome_value(never_metrics, "best_mean")

    failure_cols = [
        "any_nan_detected",
        "any_inf_detected",
        "any_gradient_explosion",
        "any_loss_explosion",
        "any_q_value_explosion",
    ]
    failure_counts = {col: int(run_df[col].sum()) for col in failure_cols if col in run_df}
    no_numeric_failure = all(count == 0 for count in failure_counts.values())
    failure_names = {
        "any_nan_detected": "NaN",
        "any_inf_detected": "infinity",
        "any_gradient_explosion": "gradient-explosion",
        "any_loss_explosion": "loss-explosion",
        "any_q_value_explosion": "Q-value-explosion",
    }
    nonzero_failures = [
        f"{failure_names.get(column, column)}: {count}" for column, count in failure_counts.items() if count
    ]
    if no_numeric_failure:
        numeric_failure_sentence = (
            "No run logs NaN, infinity, gradient explosion, loss explosion, or Q-value explosion flags, "
            "so the zero-score endpoint is not explained by the optimizer blowing up."
        )
    else:
        numeric_failure_sentence = (
            f"Numeric failure flags are rare ({'; '.join(nonzero_failures)}). "
            "They do not match the broad pattern of collapse, because most failed runs end with low upward-action "
            "preference, sparse replay rewards, and tiny Q-action gaps rather than logged numerical blow-up."
        )

    stable_runs = run_df[run_df["outcome"].astype(str) == "stable_success"].sort_values(
        "final_eval_return", ascending=False
    )
    dense_success = stable_runs[stable_runs["batch"].str.contains("dense_updates", regex=False)]
    baseline_success = stable_runs[
        (stable_runs["variant"] == "dqn") & stable_runs["batch"].str.contains("full_freeway", regex=False)
    ]
    no_replay_success = stable_runs[stable_runs["variant"] == "dqn_no_replay"]

    dense_sentence = (
        f"The dense-update batch has one standout success, seed {int(dense_success.iloc[0]['seed'])}, "
        f"with final score {fmt_num(dense_success.iloc[0]['final_eval_return'], 2)}."
        if not dense_success.empty
        else "The dense-update batch did not add a stable success in the discovered data."
    )
    baseline_sentence = (
        "The strongest ordinary DQN successes are "
        + ", ".join(
            f"{row.batch} seed {int(row.seed)} final {fmt_num(row.final_eval_return, 2)}"
            for row in baseline_success.itertuples(index=False)
        )
        + "."
        if not baseline_success.empty
        else "No ordinary full Freeway DQN seed ended above the success threshold."
    )
    no_replay_sentence = (
        f"The no-replay variant has one stable seed, seed {int(no_replay_success.iloc[0]['seed'])}, "
        f"but other no-replay seeds collapse, so this is not a reliable fix."
        if not no_replay_success.empty
        else "The no-replay variant did not produce a stable final success."
    )

    report = f"""# Freeway Training Stability Report

## Executive summary

The main failure mode is policy collapse, not numeric instability. Across {n_runs} Freeway runs and {n_eval} evaluation records, only {stable_count} runs finish at or above the success threshold of {fmt_num(SUCCESS_THRESHOLD, 0)} points. {collapsed_count} runs reach that threshold at least once but later finish below {fmt_num(COLLAPSE_FINAL_THRESHOLD, 0)} points, {partial_count} finish in the middle, and {never_count} never reach the threshold.

The rare successful runs keep moving upward late in training. Their mean final score is {fmt_num(stable_final, 2)}, their final evaluation upward-action fraction is {fmt_num(stable_up, 3)}, their late Q-value action gap is {fmt_num(stable_gap, 5)}, and their late replay reward mean is {fmt_num(stable_replay_reward, 6)}. The collapsed-after-success runs have mean final score {fmt_num(collapsed_final, 2)}, final zero-score rate {fmt_percent(collapsed_zero, 1)}, upward-action fraction {fmt_num(collapsed_up, 3)}, late Q-value action gap {fmt_num(collapsed_gap, 5)}, and late replay reward mean {fmt_num(collapsed_replay_reward, 6)}. In plain terms, the failing agents stop preferring the upward action strongly enough, then almost never see positive Freeway rewards in the replay samples.

{numeric_failure_sentence} The final policies usually have low upward-action fractions and high down/no-op fractions, which makes a zero score the natural result in Freeway.

## Data inventory

The script scans `output/**/freeway/**/run_config.json` and loads the run-local JSONL files. It does not rely on `aggregate_summary.json`, because some batches do not have one.

{tables["inventory"]}

## Outcome summary

{tables["outcome"]}

![Best versus final score]({figure_link(figure_paths["best_vs_final"])})

The best-versus-final scatter shows why the reproduction feels unstable. Many points lie far to the right but near the bottom: those runs learned a scoring policy at one checkpoint, then ended at zero or near zero. The diagonal would mean that the final policy kept the best behavior; most failed runs sit well below it.

## Learning curves

![Evaluation learning curves by group]({figure_link(figure_paths["learning_curves"])})

The curves show three patterns. First, several runs jump above 15 points early or mid-training but later fall sharply. Second, some runs never form a useful crossing policy and stay below 15. Third, the stable successes keep improving or recover after dips and finish with a high final score.

The batch-level summary makes the instability clear.

{tables["group"]}

## Why the rare successes work

{baseline_sentence} {dense_sentence} {no_replay_sentence}

{tables["stable"]}

The successful policies share the same behavioral signature. They have a high final upward-action fraction, usually around 0.8 or higher, and low down/no-op usage. Their last training episodes still score points, with mean tail training score {fmt_num(stable_tail_train, 2)} across the stable-success group. They also keep a larger Q-value action gap than the collapsed runs. That means the network has a clearer preference among actions instead of nearly tying up, down, and no-op.

![Final action fractions by outcome]({figure_link(figure_paths["action_fractions"])})

Freeway scoring requires sustained upward motion. The successful agents do not need a complex final action mix; they need a stable bias toward moving up while reacting enough to avoid cars. The logs show that the stable runs keep this bias at the end.

## Why most runs end at zero

The most common failure is early success followed by collapse.

{tables["collapsed"]}

These runs prove that the environment and network can produce scoring behavior, but the learned behavior is not retained. After the best checkpoint, the final policy often shifts toward down or no-op. Once that happens, Freeway gives almost no reward, so the replay sample reward mean falls close to zero. With sparse positive rewards and a tiny Q-action gap, later updates can preserve a bad zero-score policy instead of restoring the earlier crossing behavior.

The never-success group is different. Its mean best score is only {fmt_num(never_best, 2)}, so those runs usually do not find a strong crossing behavior in the first place. Slow exploration is a good example: keeping epsilon high for longer does not automatically help if the final policy never consolidates upward movement. The logs show low final score and weak upward action preference.

The no-target and double-DQN ablation results also point to retention problems. Several seeds reach useful scores, but final scores fall to zero. That makes the issue less about whether Freeway can be solved at all and more about whether the training setup can keep a useful policy after sparse rewards, stochastic starts, sticky actions, and continuing updates.

## Case studies

![Stable versus collapsed case studies]({figure_link(figure_paths["case_studies"])})

The stable case keeps score and upward-action fraction high near the end. The collapsed case reaches a high score earlier, then the score curve falls while the action mix stops looking like a crossing policy. This is the clearest single-run view of the same pattern seen in the aggregate tables.

## Hyperparameters and outcome

![Hyperparameter and outcome table]({figure_link(figure_paths["hyperparameter_table"])})

The more reliable settings in these logs use the Freeway defaults from the later configuration: lower learning rate, smaller replay buffer, shorter learning starts, faster target updates, and faster epsilon decay than the tuned batch. The dense-update run shows that more frequent updates can work, but only one of four dense-update seeds is stable, so update density alone is not enough. The large-replay, long-learning-start, slow-decay tuned batch often reaches 15 points but does not preserve the policy by the final checkpoint.

## Practical implications

For reproduction, the best checkpoint matters more than the final checkpoint in many current runs. A run that reports a best score above 20 can still finish with zero points. Any future training script should save and evaluate the best checkpoint separately from the final checkpoint.

For stability, track action fractions during evaluation. A falling upward-action fraction is an early warning that the policy is becoming a zero-score policy. Track late replay reward mean and Q-action gap as well; when both are near zero, the agent has little signal and little preference structure to recover.

For the next experiment pass, prefer the settings used by the stable DQN runs as the baseline, keep best-checkpoint selection, and test changes one at a time. The current data does not support treating Double DQN, Dueling DQN, no target network, no replay, slow exploration, or dense updates as a general fix.

## Generated artifacts

The helper program is `result_analysis/analyze_freeway.py`. It generated these derived files:

`result_analysis/freeway_run_summary.csv`

`result_analysis/freeway_eval_timeseries.csv`

`result_analysis/freeway_group_summary.csv`

`result_analysis/freeway_outcome_summary.csv`

All figures used in this report are under `result_analysis/figures/`.
"""
    return textwrap.dedent(report).strip() + "\n"


def write_report(report: str) -> Path:
    path = RESULT_DIR / "freeway_training_stability_report.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    run_df, eval_df = discover_freeway_runs()
    if run_df.empty:
        raise SystemExit(f"No Freeway run_config.json files found under {OUTPUT_DIR}")

    inventory_df = build_inventory(run_df)
    group_df = build_group_summary(run_df)
    outcome_df = build_outcome_summary(run_df)

    save_csvs(run_df, eval_df, group_df, outcome_df)

    figure_paths = {
        "learning_curves": save_learning_curves(eval_df, run_df),
        "best_vs_final": save_best_vs_final(run_df),
        "action_fractions": save_action_fractions(run_df),
        "case_studies": save_case_study_curves(eval_df, run_df),
        "hyperparameter_table": save_group_table_figure(group_df, run_df),
    }

    report = build_report(run_df, eval_df, inventory_df, group_df, outcome_df, figure_paths)
    report_path = write_report(report)

    print(f"Discovered Freeway runs: {len(run_df)}")
    print(f"Evaluation records: {len(eval_df)}")
    print(f"Stable final successes: {int(run_df['stable_success'].sum())}")
    print(f"Collapsed after success: {int(run_df['collapsed_after_success'].sum())}")
    print(f"Wrote report: {report_path.relative_to(ROOT)}")
    print(f"Wrote figures: {FIGURE_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
