#!/usr/bin/env python3
"""Generate Freeway up-biased DQN analysis tables, figures, and reports."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rl-analysis-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "dqn",
    "double_dqn",
    "dueling_dqn",
    "double_dueling_dqn",
]

VARIANT_LABELS = {
    "dqn": "DQN baseline",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "double_dueling_dqn": "Double + Dueling DQN",
}

SHORT_LABELS = {
    "dqn": "DQN",
    "double_dqn": "Double",
    "dueling_dqn": "Dueling",
    "double_dueling_dqn": "Double+\nDueling",
}

VARIANT_NOTES = {
    "dqn": "Baseline DQN with experience replay and a periodically copied target network.",
    "double_dqn": "Double DQN uses the online network to select the next action and the target network to evaluate it.",
    "dueling_dqn": "Dueling DQN splits the network head into state-value and action-advantage streams.",
    "double_dueling_dqn": "Double + Dueling DQN combines Double DQN target selection with the dueling architecture.",
}

COLORS = {
    "dqn": "#2E86AB",
    "double_dqn": "#2A9D8F",
    "dueling_dqn": "#F4A261",
    "double_dueling_dqn": "#6D597A",
}

SCORE_THRESHOLDS = [5.0, 10.0, 15.0, 22.5]
SUCCESS_THRESHOLD = 15.0
REFERENCE_RANDOM = 0.0
REFERENCE_HUMAN = 29.6
REFERENCE_DQN = 30.3

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="output/run_freeway_upbiased_20260520_110511",
        help="Root containing the freeway output directory.",
    )
    parser.add_argument(
        "--analysis-dir",
        default="analysis/freeway_upbiased_20260520_110511",
        help="Directory for generated analysis artifacts.",
    )
    parser.add_argument("--env-slug", default="freeway", help="Environment slug under output-root.")
    parser.add_argument("--report-name", default="freeway_upbiased_report.md")
    parser.add_argument("--report-title", default="Freeway Up-Biased DQN Report")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_seed_runs(env_dir: Path, variant: str) -> list[Path]:
    variant_dir = env_dir / variant
    run_dirs: list[Path] = []
    for seed_dir in sorted(variant_dir.glob("seed_*")):
        candidates = [
            path
            for path in sorted(seed_dir.iterdir())
            if path.is_dir() and (path / "summary.json").exists()
        ]
        if candidates:
            run_dirs.append(max(candidates, key=lambda path: path.stat().st_mtime))
    return run_dirs


def get_nested(data: dict, path: str, default=np.nan):
    node = data
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    if node is None:
        return default
    return node


def to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def ci95(series: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(arr)
    if n <= 1:
        return float("nan")
    std = float(np.std(arr, ddof=1))
    tcrit = T_CRIT_95.get(n - 1, 1.96)
    return float(tcrit * std / math.sqrt(n))


def safe_mean(series: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return float("nan")
    return float(np.mean(arr))


def safe_std(series: Iterable[float]) -> float:
    arr = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def threshold_key(threshold: float) -> str:
    return str(float(threshold))


def threshold_col(threshold: float) -> str:
    return str(float(threshold)).replace(".", "_")


def format_number(value: float, digits: int = 1, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(numeric) or math.isinf(numeric):
        return na
    return f"{numeric:.{digits}f}"


def format_signed(value: float, digits: int = 1, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(numeric) or math.isinf(numeric):
        return na
    return f"{numeric:+.{digits}f}"


def format_percent(value: float, digits: int = 0, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(numeric) or math.isinf(numeric):
        return na
    return f"{100.0 * numeric:.{digits}f}%"


def format_steps(value: float, na: str = "n/a") -> str:
    if value is None:
        return na
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return na
    if math.isnan(numeric) or math.isinf(numeric):
        return na
    return f"{int(round(numeric)):,}"


def mean_ci_text(mean: float, ci: float, digits: int = 1) -> str:
    if math.isnan(float(mean)):
        return "n/a"
    if ci is None or math.isnan(float(ci)):
        return format_number(mean, digits)
    return f"{format_number(mean, digits)} +/- {format_number(ci, digits)}"


def count_text(count: int, total: int, noun: str = "seed") -> str:
    if count == 0:
        return f"no {noun}s"
    if count == total:
        return f"all {total} {noun}s"
    return f"{count}/{total} {noun}s"


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def cell(value: object) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def load_data(output_root: Path, env_slug: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    env_dir = output_root / env_slug
    summary_rows: list[dict] = []
    eval_frames: list[pd.DataFrame] = []
    update_frames: list[pd.DataFrame] = []

    for variant in VARIANT_ORDER:
        for run_dir in latest_seed_runs(env_dir, variant):
            summary = read_json(run_dir / "summary.json")
            if not summary.get("training_completed", False):
                continue
            config = read_json(run_dir / "run_config.json") if (run_dir / "run_config.json").exists() else {}
            performance = summary.get("performance", {})
            sample_efficiency = summary.get("sample_efficiency", {})
            stability = summary.get("stability", {})
            optimization = summary.get("optimization", {})
            q_diagnostics = summary.get("q_diagnostics", {})
            replay = summary.get("replay_diagnostics", {})
            target = summary.get("target_network_diagnostics", {})
            failure = summary.get("failure_diagnostics", {})
            compute = summary.get("compute", {})
            freeway = summary.get("freeway", {})

            final_score = to_float(performance.get("final_eval_return_mean"))
            success_threshold = to_float(sample_efficiency.get("threshold", SUCCESS_THRESHOLD))
            threshold_steps = freeway.get("first_step_reaching_score_thresholds", {})
            row = {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": int(summary["seed"]),
                "run_id": summary.get("run_id", run_dir.name),
                "run_dir": str(run_dir),
                "training_completed": bool(summary.get("training_completed", False)),
                "total_env_steps": to_float(summary.get("total_env_steps")),
                "total_ale_frames": to_float(summary.get("total_ale_frames")),
                "total_updates": to_float(summary.get("total_updates")),
                "total_episodes": to_float(summary.get("total_episodes")),
                "wall_time_total_sec": to_float(compute.get("wall_time_total_sec")),
                "env_steps_per_second": to_float(compute.get("env_steps_per_second")),
                "updates_per_second": to_float(compute.get("updates_per_second")),
                "final_score": final_score,
                "final_score_median": to_float(performance.get("final_eval_return_median")),
                "final_score_std_within_eval": to_float(performance.get("final_eval_return_std")),
                "best_score": to_float(performance.get("best_eval_return_mean")),
                "best_eval_step": to_float(performance.get("best_eval_step")),
                "area_under_eval_curve": to_float(performance.get("area_under_eval_curve")),
                "normalized_auc": to_float(performance.get("normalized_area_under_eval_curve")),
                "final_train_score_mean_last_100": to_float(performance.get("final_train_return_mean_last_100")),
                "final_train_score_std_last_100": to_float(performance.get("final_train_return_std_last_100")),
                "success_threshold": success_threshold,
                "first_reach_step": to_float(sample_efficiency.get("first_step_reaching_threshold")),
                "first_sustained_step": to_float(sample_efficiency.get("first_step_sustained_threshold")),
                "reached_threshold": not bool(sample_efficiency.get("never_reached_threshold", True)),
                "collapse_count": to_float(stability.get("catastrophic_collapse_count")),
                "largest_eval_drop": to_float(stability.get("largest_eval_drop")),
                "largest_eval_drop_fraction": to_float(stability.get("largest_eval_drop_fraction")),
                "eval_score_std_across_time": to_float(stability.get("eval_return_std_across_time")),
                "rolling_score_std_mean": to_float(stability.get("rolling_return_std_mean")),
                "td_loss_last10": to_float(optimization.get("td_loss_mean_last_10pct")),
                "td_error_abs_mean_last10": to_float(optimization.get("td_error_abs_mean_last_10pct")),
                "td_error_abs_p95_last10": to_float(optimization.get("td_error_abs_p95_last_10pct")),
                "grad_norm_mean_last10": to_float(optimization.get("grad_norm_mean_last_10pct")),
                "grad_norm_max": to_float(optimization.get("grad_norm_max")),
                "gradient_clip_fraction": to_float(optimization.get("gradient_clip_fraction")),
                "q_overestimation_proxy_last10": to_float(
                    q_diagnostics.get("mean_q_overestimation_proxy_last_10pct")
                ),
                "online_q_mean_last10": to_float(q_diagnostics.get("online_q_mean_last_10pct")),
                "online_q_max_mean_last10": to_float(q_diagnostics.get("online_q_max_mean_last_10pct")),
                "target_q_mean_last10": to_float(q_diagnostics.get("target_q_mean_last_10pct")),
                "max_q_value_seen": to_float(q_diagnostics.get("max_q_value_seen")),
                "replay_enabled": bool(replay.get("replay_enabled", False)),
                "final_buffer_size": to_float(replay.get("final_buffer_size")),
                "sample_age_mean_last10": to_float(replay.get("sample_age_mean_last_10pct")),
                "sample_consecutive_transition_fraction_last10": to_float(
                    replay.get("sample_consecutive_transition_fraction_last_10pct")
                ),
                "target_network_enabled": bool(target.get("target_network_enabled", False)),
                "target_update_count": to_float(target.get("target_update_count")),
                "online_target_param_l2_mean": to_float(target.get("online_target_param_l2_mean")),
                "online_target_param_l2_max": to_float(target.get("online_target_param_l2_max")),
                "nan_detected": bool(failure.get("nan_detected", False)),
                "inf_detected": bool(failure.get("inf_detected", False)),
                "loss_explosion": bool(failure.get("loss_explosion", False)),
                "q_value_explosion": bool(failure.get("q_value_explosion", False)),
                "gradient_explosion": bool(failure.get("gradient_explosion", False)),
                "early_terminated": bool(failure.get("early_terminated", False)),
                "final_zero_score_rate": to_float(freeway.get("final_zero_score_rate")),
                "reference_random_score": to_float(freeway.get("reference_random_score", REFERENCE_RANDOM)),
                "reference_human_score": to_float(freeway.get("reference_human_score", REFERENCE_HUMAN)),
                "reference_dqn_score": to_float(freeway.get("reference_dqn_score", REFERENCE_DQN)),
                "use_target_network": bool(get_nested(config, "algorithm.use_target_network", False)),
                "use_replay_buffer": bool(get_nested(config, "algorithm.use_replay_buffer", False)),
                "use_double_dqn": bool(get_nested(config, "algorithm.use_double_dqn", False)),
                "use_dueling_network": bool(get_nested(config, "algorithm.use_dueling_network", False)),
                "freeway_up_bias": to_float(get_nested(config, "exploration.freeway_up_bias")),
                "learning_rate": to_float(get_nested(config, "optimizer.learning_rate")),
                "gamma": to_float(get_nested(config, "dqn_hyperparameters.gamma")),
                "batch_size": to_float(get_nested(config, "dqn_hyperparameters.batch_size")),
                "target_update_frequency": to_float(
                    get_nested(config, "dqn_hyperparameters.target_update_frequency_env_steps")
                ),
                "epsilon_decay_steps": to_float(get_nested(config, "exploration.epsilon_decay_env_steps")),
            }
            for threshold in SCORE_THRESHOLDS:
                key = threshold_key(threshold)
                col = threshold_col(threshold)
                first_step = to_float(threshold_steps.get(key))
                row[f"first_reach_score_{col}_step"] = first_step
                row[f"reached_score_{col}"] = not math.isnan(first_step)
            row["best_final_gap"] = row["best_score"] - row["final_score"]
            row["final_solved"] = bool(row["final_score"] >= row["success_threshold"])
            summary_rows.append(row)

            eval_rows = read_jsonl(run_dir / "eval_metrics.jsonl")
            if eval_rows:
                eval_frame = pd.json_normalize(eval_rows, sep=".")
                eval_frame["variant"] = variant
                eval_frame["variant_label"] = VARIANT_LABELS.get(variant, variant)
                eval_frame["seed"] = int(summary["seed"])
                eval_frame["run_id"] = summary.get("run_id", run_dir.name)
                eval_frame["run_dir"] = str(run_dir)
                eval_frames.append(eval_frame)

            update_rows = read_jsonl(run_dir / "train_update_metrics.jsonl")
            if update_rows:
                update_frame = pd.json_normalize(update_rows, sep=".")
                update_frame["variant"] = variant
                update_frame["variant_label"] = VARIANT_LABELS.get(variant, variant)
                update_frame["seed"] = int(summary["seed"])
                update_frame["run_id"] = summary.get("run_id", run_dir.name)
                update_frame["run_dir"] = str(run_dir)
                update_frames.append(update_frame)

    summary_df = pd.DataFrame(summary_rows)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    update_df = (
        pd.concat([frame.dropna(axis=1, how="all") for frame in update_frames], ignore_index=True)
        if update_frames
        else pd.DataFrame()
    )

    if summary_df.empty:
        raise RuntimeError(f"No completed summaries found under {env_dir}.")

    summary_df["variant"] = pd.Categorical(summary_df["variant"], VARIANT_ORDER, ordered=True)
    summary_df = summary_df.sort_values(["variant", "seed"]).reset_index(drop=True)
    if not eval_df.empty:
        eval_df["variant"] = pd.Categorical(eval_df["variant"], VARIANT_ORDER, ordered=True)
        eval_df = eval_df.sort_values(["variant", "seed", "global_env_step"]).reset_index(drop=True)
    if not update_df.empty:
        update_df["variant"] = pd.Categorical(update_df["variant"], VARIANT_ORDER, ordered=True)
        update_df = update_df.sort_values(["variant", "seed", "global_env_step"]).reset_index(drop=True)
    return summary_df, eval_df, update_df


def build_variant_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for variant in VARIANT_ORDER:
        group = summary_df[summary_df["variant"] == variant]
        if group.empty:
            continue
        reached = group["reached_threshold"].astype(float)
        solved = group["final_solved"].astype(float)
        record = {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "n": int(len(group)),
            "final_score_mean": safe_mean(group["final_score"]),
            "final_score_std": safe_std(group["final_score"]),
            "final_score_ci95": ci95(group["final_score"]),
            "final_score_median": float(group["final_score"].median()),
            "final_score_min": float(group["final_score"].min()),
            "final_score_max": float(group["final_score"].max()),
            "final_solved_rate": safe_mean(solved),
            "final_solved_count": int(group["final_solved"].sum()),
            "best_score_mean": safe_mean(group["best_score"]),
            "best_score_std": safe_std(group["best_score"]),
            "best_score_ci95": ci95(group["best_score"]),
            "normalized_auc_mean": safe_mean(group["normalized_auc"]),
            "normalized_auc_std": safe_std(group["normalized_auc"]),
            "normalized_auc_ci95": ci95(group["normalized_auc"]),
            "threshold_reach_rate": safe_mean(reached),
            "threshold_reach_count": int(group["reached_threshold"].sum()),
            "first_reach_step_median": float(group["first_reach_step"].dropna().median())
            if group["first_reach_step"].notna().any()
            else float("nan"),
            "first_reach_step_mean": safe_mean(group["first_reach_step"]),
            "first_sustained_step_median": float(group["first_sustained_step"].dropna().median())
            if group["first_sustained_step"].notna().any()
            else float("nan"),
            "collapse_count_mean": safe_mean(group["collapse_count"]),
            "collapse_count_ci95": ci95(group["collapse_count"]),
            "largest_eval_drop_mean": safe_mean(group["largest_eval_drop"]),
            "eval_score_std_across_time_mean": safe_mean(group["eval_score_std_across_time"]),
            "best_final_gap_mean": safe_mean(group["best_final_gap"]),
            "best_final_gap_ci95": ci95(group["best_final_gap"]),
            "td_loss_last10_mean": safe_mean(group["td_loss_last10"]),
            "td_loss_last10_ci95": ci95(group["td_loss_last10"]),
            "td_error_abs_p95_last10_mean": safe_mean(group["td_error_abs_p95_last10"]),
            "td_error_abs_p95_last10_ci95": ci95(group["td_error_abs_p95_last10"]),
            "gradient_clip_fraction_mean": safe_mean(group["gradient_clip_fraction"]),
            "gradient_clip_fraction_ci95": ci95(group["gradient_clip_fraction"]),
            "q_overestimation_proxy_last10_mean": safe_mean(group["q_overestimation_proxy_last10"]),
            "q_overestimation_proxy_last10_ci95": ci95(group["q_overestimation_proxy_last10"]),
            "max_q_value_seen_mean": safe_mean(group["max_q_value_seen"]),
            "sample_age_mean_last10_mean": safe_mean(group["sample_age_mean_last10"]),
            "sample_consecutive_transition_fraction_last10_mean": safe_mean(
                group["sample_consecutive_transition_fraction_last10"]
            ),
            "online_target_param_l2_mean": safe_mean(group["online_target_param_l2_mean"]),
            "env_steps_per_second_mean": safe_mean(group["env_steps_per_second"]),
            "wall_time_total_sec_mean": safe_mean(group["wall_time_total_sec"]),
            "final_zero_score_rate_mean": safe_mean(group["final_zero_score_rate"]),
            "final_zero_score_rate_ci95": ci95(group["final_zero_score_rate"]),
            "failure_rate": safe_mean(
                group[
                    [
                        "nan_detected",
                        "inf_detected",
                        "loss_explosion",
                        "q_value_explosion",
                        "gradient_explosion",
                        "early_terminated",
                    ]
                ].any(axis=1).astype(float)
            ),
        }
        for threshold in SCORE_THRESHOLDS:
            col = threshold_col(threshold)
            reached_col = f"reached_score_{col}"
            first_col = f"first_reach_score_{col}_step"
            record[f"score_{col}_reach_rate"] = safe_mean(group[reached_col].astype(float))
            record[f"score_{col}_reach_count"] = int(group[reached_col].sum())
            record[f"score_{col}_first_reach_step_median"] = (
                float(group[first_col].dropna().median()) if group[first_col].notna().any() else float("nan")
            )
        records.append(record)
    return pd.DataFrame(records)


def build_eval_curve(eval_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (variant, step), group in eval_df.groupby(["variant", "global_env_step"], observed=True):
        if pd.isna(variant):
            continue
        record = {
            "variant": str(variant),
            "variant_label": VARIANT_LABELS[str(variant)],
            "global_env_step": int(step),
            "n": int(len(group)),
            "score_mean": safe_mean(group["return_mean"]),
            "score_std": safe_std(group["return_mean"]),
            "score_ci95": ci95(group["return_mean"]),
            "zero_score_rate_mean": safe_mean(group.get("freeway_eval.zero_score_rate", pd.Series(dtype=float))),
            "action_up_fraction_mean": safe_mean(
                group.get("freeway_eval.action_up_fraction_mean", pd.Series(dtype=float))
            ),
            "action_down_fraction_mean": safe_mean(
                group.get("freeway_eval.action_down_fraction_mean", pd.Series(dtype=float))
            ),
            "action_noop_fraction_mean": safe_mean(
                group.get("freeway_eval.action_noop_fraction_mean", pd.Series(dtype=float))
            ),
            "episode_length_mean": safe_mean(group.get("episode_length_mean", pd.Series(dtype=float))),
            "action_entropy_mean": safe_mean(group.get("action_entropy_mean", pd.Series(dtype=float))),
        }
        for threshold in SCORE_THRESHOLDS:
            key = f"freeway_eval.score_threshold_success_rates.{threshold_key(threshold)}"
            col = threshold_col(threshold)
            record[f"score_{col}_success_rate_mean"] = safe_mean(group.get(key, pd.Series(dtype=float)))
        records.append(record)
    curve = pd.DataFrame(records)
    curve["variant"] = pd.Categorical(curve["variant"], VARIANT_ORDER, ordered=True)
    return curve.sort_values(["variant", "global_env_step"]).reset_index(drop=True)


def build_final_behavior(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    last_eval = (
        eval_df.sort_values(["variant", "seed", "global_env_step"])
        .groupby(["variant", "seed"], observed=True)
        .tail(1)
        .reset_index(drop=True)
    )
    metrics = {
        "final_score": "return_mean",
        "score_std_within_eval": "return_std",
        "zero_score_rate": "freeway_eval.zero_score_rate",
        "action_up_fraction": "freeway_eval.action_up_fraction_mean",
        "action_down_fraction": "freeway_eval.action_down_fraction_mean",
        "action_noop_fraction": "freeway_eval.action_noop_fraction_mean",
        "episode_length": "episode_length_mean",
        "action_entropy": "action_entropy_mean",
    }
    for threshold in SCORE_THRESHOLDS:
        metrics[f"score_{threshold_col(threshold)}_success_rate"] = (
            f"freeway_eval.score_threshold_success_rates.{threshold_key(threshold)}"
        )

    records: list[dict] = []
    for variant in VARIANT_ORDER:
        group = last_eval[last_eval["variant"] == variant]
        if group.empty:
            continue
        record = {"variant": variant, "variant_label": VARIANT_LABELS[variant], "n": int(len(group))}
        for out_col, in_col in metrics.items():
            values = group[in_col] if in_col in group.columns else pd.Series(dtype=float)
            record[f"{out_col}_mean"] = safe_mean(values)
            record[f"{out_col}_std"] = safe_std(values)
            record[f"{out_col}_ci95"] = ci95(values)
        records.append(record)
    return pd.DataFrame(records)


def build_phase_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    if eval_df.empty:
        return pd.DataFrame()
    phases = eval_df.copy()
    phases["phase"] = np.select(
        [
            (phases["global_env_step"] >= 10_000) & (phases["global_env_step"] <= 250_000),
            (phases["global_env_step"] > 250_000) & (phases["global_env_step"] <= 600_000),
            phases["global_env_step"] > 600_000,
        ],
        ["early: 10k-250k", "middle: 260k-600k", "late: 610k-1M"],
        default="unknown",
    )
    phases = phases[phases["phase"] != "unknown"]
    per_seed = (
        phases.groupby(["variant", "seed", "phase"], observed=True)["return_mean"]
        .mean()
        .reset_index(name="phase_score_mean")
    )
    records: list[dict] = []
    for (variant, phase), group in per_seed.groupby(["variant", "phase"], observed=True):
        records.append(
            {
                "variant": str(variant),
                "variant_label": VARIANT_LABELS[str(variant)],
                "phase": str(phase),
                "n": int(len(group)),
                "score_mean": safe_mean(group["phase_score_mean"]),
                "score_std": safe_std(group["phase_score_mean"]),
                "score_ci95": ci95(group["phase_score_mean"]),
            }
        )
    phase_df = pd.DataFrame(records)
    phase_order = ["early: 10k-250k", "middle: 260k-600k", "late: 610k-1M"]
    phase_df["variant"] = pd.Categorical(phase_df["variant"], VARIANT_ORDER, ordered=True)
    phase_df["phase"] = pd.Categorical(phase_df["phase"], phase_order, ordered=True)
    return phase_df.sort_values(["variant", "phase"]).reset_index(drop=True)


def build_paired_comparison(summary_df: pd.DataFrame, baseline: str = "dqn") -> pd.DataFrame:
    base = summary_df[summary_df["variant"] == baseline].set_index("seed")
    records: list[dict] = []
    for variant in VARIANT_ORDER:
        if variant == baseline:
            continue
        other = summary_df[summary_df["variant"] == variant].set_index("seed")
        shared = sorted(set(base.index) & set(other.index))
        if not shared:
            continue
        b = base.loc[shared]
        o = other.loc[shared]
        final_delta = o["final_score"].to_numpy(dtype=float) - b["final_score"].to_numpy(dtype=float)
        best_delta = o["best_score"].to_numpy(dtype=float) - b["best_score"].to_numpy(dtype=float)
        auc_delta = o["normalized_auc"].to_numpy(dtype=float) - b["normalized_auc"].to_numpy(dtype=float)

        reach_mask = o["first_reach_step"].notna().to_numpy() & b["first_reach_step"].notna().to_numpy()
        reach_delta = (
            o["first_reach_step"].to_numpy(dtype=float)[reach_mask]
            - b["first_reach_step"].to_numpy(dtype=float)[reach_mask]
            if reach_mask.any()
            else np.array([], dtype=float)
        )
        collapse_delta = o["collapse_count"].to_numpy(dtype=float) - b["collapse_count"].to_numpy(dtype=float)
        records.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "n_shared_seeds": len(shared),
                "final_delta_mean": float(np.mean(final_delta)),
                "final_delta_ci95": ci95(final_delta),
                "final_delta_median": float(np.median(final_delta)),
                "final_win_rate_vs_baseline": float(np.mean(final_delta > 0.0)),
                "best_delta_mean": float(np.mean(best_delta)),
                "best_delta_ci95": ci95(best_delta),
                "normalized_auc_delta_mean": float(np.mean(auc_delta)),
                "normalized_auc_delta_ci95": ci95(auc_delta),
                "first_reach_step_delta_mean": float(np.mean(reach_delta)) if len(reach_delta) else float("nan"),
                "first_reach_step_delta_ci95": ci95(reach_delta) if len(reach_delta) else float("nan"),
                "paired_reach_count": int(len(reach_delta)),
                "collapse_delta_mean": float(np.mean(collapse_delta)),
                "collapse_delta_ci95": ci95(collapse_delta),
            }
        )
    return pd.DataFrame(records)


def save_tables(
    tables_dir: Path,
    summary_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    paired: pd.DataFrame,
    eval_curve: pd.DataFrame,
    final_behavior: pd.DataFrame,
    phase_summary: pd.DataFrame,
) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(tables_dir / "freeway_run_summary.csv", index=False)
    variant_summary.to_csv(tables_dir / "freeway_variant_summary.csv", index=False)
    paired.to_csv(tables_dir / "freeway_paired_vs_dqn.csv", index=False)
    eval_curve.to_csv(tables_dir / "freeway_eval_curve_summary.csv", index=False)
    final_behavior.to_csv(tables_dir / "freeway_final_behavior_summary.csv", index=False)
    phase_summary.to_csv(tables_dir / "freeway_phase_score_summary.csv", index=False)


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (9, 5.2),
            "figure.dpi": 140,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
        }
    )


def ordered_variants(df: pd.DataFrame) -> list[str]:
    present = set(str(v) for v in df["variant"].dropna().unique())
    return [variant for variant in VARIANT_ORDER if variant in present]


def bar_with_ci(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_col: str,
    ci_col: str | None,
    ylabel: str,
    title: str,
    percent: bool = False,
) -> None:
    variants = ordered_variants(df)
    means = [float(df.loc[df["variant"] == variant, mean_col].iloc[0]) for variant in variants]
    errs = None
    if ci_col is not None and ci_col in df.columns:
        errs = [
            0.0
            if pd.isna(df.loc[df["variant"] == variant, ci_col].iloc[0])
            else float(df.loc[df["variant"] == variant, ci_col].iloc[0])
            for variant in variants
        ]
    if percent:
        means = [100.0 * value for value in means]
        if errs is not None:
            errs = [100.0 * value for value in errs]
    x = np.arange(len(variants))
    ax.bar(x, means, yerr=errs, capsize=3 if errs else 0, color=[COLORS[v] for v in variants], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[v] for v in variants], rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if percent:
        ax.set_ylim(bottom=0)


def add_freeway_reference_lines(ax: plt.Axes, include_success: bool = True) -> None:
    if include_success:
        ax.axhline(SUCCESS_THRESHOLD, color="#333333", linewidth=1.1, linestyle="--", label="score 15")
    ax.axhline(REFERENCE_HUMAN, color="#8B1E3F", linewidth=1.0, linestyle=":", label="human 29.6")
    ax.axhline(REFERENCE_DQN, color="#4A4A4A", linewidth=1.0, linestyle="-.", label="DQN ref 30.3")


def plot_learning_curves(eval_curve: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for variant in ordered_variants(eval_curve):
        curve = eval_curve[eval_curve["variant"] == variant]
        x = curve["global_env_step"].to_numpy(dtype=float)
        y = curve["score_mean"].to_numpy(dtype=float)
        ci = curve["score_ci95"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, label=SHORT_LABELS[variant].replace("\n", " "), color=COLORS[variant], linewidth=2)
        ax.fill_between(x, y - ci, y + ci, color=COLORS[variant], alpha=0.12, linewidth=0)
    add_freeway_reference_lines(ax)
    ax.axhline(REFERENCE_RANDOM, color="#999999", linewidth=0.8, linestyle="--", label="random 0")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation score mean")
    ax.set_title("Freeway evaluation learning curves across seeds")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_eval_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_score(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    variants = ordered_variants(summary_df)
    data = [summary_df[summary_df["variant"] == variant]["final_score"].to_numpy(dtype=float) for variant in variants]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    boxes = ax.boxplot(data, patch_artist=True, tick_labels=[SHORT_LABELS[v] for v in variants], showfliers=False)
    for patch, variant in zip(boxes["boxes"], variants):
        patch.set_facecolor(COLORS[variant])
        patch.set_alpha(0.32)
        patch.set_edgecolor(COLORS[variant])
    rng = np.random.default_rng(7)
    for index, (variant, values) in enumerate(zip(variants, data), start=1):
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(np.full(len(values), index) + jitter, values, color=COLORS[variant], s=32, zorder=3)
    add_freeway_reference_lines(ax)
    ax.set_ylabel("Final evaluation score")
    seed_count = int(summary_df.groupby("variant", observed=True)["seed"].nunique().max())
    ax.set_title(f"Final Freeway score distribution across {seed_count} seeds")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_final_score_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_paired_deltas(paired: pd.DataFrame, figures_dir: Path) -> None:
    variants = [variant for variant in VARIANT_ORDER if variant != "dqn" and variant in set(paired["variant"])]
    df = paired.set_index("variant").loc[variants].reset_index()
    y = np.arange(len(df))
    means = df["final_delta_mean"].to_numpy(dtype=float)
    errs = df["final_delta_ci95"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(y, means, xerr=errs, capsize=3, color=[COLORS[v] for v in df["variant"]], alpha=0.9)
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT_LABELS[v].replace("\n", " ") for v in df["variant"]])
    ax.set_xlabel("Final score delta vs DQN, same seed")
    ax.set_title("Paired seed comparison against DQN")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_paired_final_delta_vs_dqn.png", bbox_inches="tight")
    plt.close(fig)


def plot_sample_efficiency(variant_summary: pd.DataFrame, summary_df: pd.DataFrame, figures_dir: Path) -> None:
    variants = ordered_variants(variant_summary)
    total_steps = int(summary_df["total_env_steps"].dropna().max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bar_with_ci(
        axes[0],
        variant_summary,
        "normalized_auc_mean",
        "normalized_auc_ci95",
        "Average eval score over training",
        "Normalized AUC",
    )
    axes[0].axhline(0, color="#333333", linewidth=1)

    x = np.arange(len(variants))
    heights: list[float] = []
    colors: list[str] = []
    for variant in variants:
        row = variant_summary[variant_summary["variant"] == variant].iloc[0]
        if row["threshold_reach_count"] > 0:
            heights.append(float(row["first_reach_step_median"]))
            colors.append(COLORS[variant])
        else:
            heights.append(float(total_steps))
            colors.append("#BBBBBB")
    bars = axes[1].bar(x, heights, color=colors, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT_LABELS[v] for v in variants])
    axes[1].set_ylabel("Median first step reaching score >= 15")
    axes[1].set_title("Sample efficiency and reach rate")
    axes[1].set_ylim(0, total_steps * 1.14)
    for bar, variant in zip(bars, variants):
        row = variant_summary[variant_summary["variant"] == variant].iloc[0]
        label = f"{int(row['threshold_reach_count'])}/{int(row['n'])}"
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total_steps * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_sample_efficiency_auc.png", bbox_inches="tight")
    plt.close(fig)


def plot_stability(variant_summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    bar_with_ci(
        axes[0],
        variant_summary,
        "collapse_count_mean",
        "collapse_count_ci95",
        "Collapse count per run",
        "Catastrophic collapse count",
    )
    bar_with_ci(
        axes[1],
        variant_summary,
        "best_final_gap_mean",
        "best_final_gap_ci95",
        "Best score minus final score",
        "Late-training regression",
    )
    axes[1].axhline(0, color="#333333", linewidth=1)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_stability_regression.png", bbox_inches="tight")
    plt.close(fig)


def plot_optimization(variant_summary: pd.DataFrame, figures_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
    bar_with_ci(
        axes[0, 0],
        variant_summary,
        "td_loss_last10_mean",
        "td_loss_last10_ci95",
        "TD loss",
        "Late TD loss",
    )
    bar_with_ci(
        axes[0, 1],
        variant_summary,
        "td_error_abs_p95_last10_mean",
        "td_error_abs_p95_last10_ci95",
        "P95 absolute TD error",
        "Tail TD error",
    )
    bar_with_ci(
        axes[1, 0],
        variant_summary,
        "q_overestimation_proxy_last10_mean",
        "q_overestimation_proxy_last10_ci95",
        "Online Q max minus target Q mean",
        "Q overestimation proxy",
    )
    axes[1, 0].axhline(0, color="#333333", linewidth=1)
    bar_with_ci(
        axes[1, 1],
        variant_summary,
        "gradient_clip_fraction_mean",
        "gradient_clip_fraction_ci95",
        "Updates clipped (%)",
        "Gradient clipping fraction",
        percent=True,
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_optimization_q_diagnostics.png", bbox_inches="tight")
    plt.close(fig)


def plot_behavior(final_behavior: pd.DataFrame, figures_dir: Path) -> None:
    if final_behavior.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3))
    bar_with_ci(
        axes[0, 0],
        final_behavior,
        "score_15_0_success_rate_mean",
        "score_15_0_success_rate_ci95",
        "Rate (%)",
        "Final eval score >= 15",
        percent=True,
    )
    bar_with_ci(
        axes[0, 1],
        final_behavior,
        "score_22_5_success_rate_mean",
        "score_22_5_success_rate_ci95",
        "Rate (%)",
        "Final eval score >= 22.5",
        percent=True,
    )
    bar_with_ci(
        axes[1, 0],
        final_behavior,
        "zero_score_rate_mean",
        "zero_score_rate_ci95",
        "Rate (%)",
        "Final zero-score episodes",
        percent=True,
    )

    variants = ordered_variants(final_behavior)
    x = np.arange(len(variants))
    width = 0.25
    up = [float(final_behavior.loc[final_behavior["variant"] == v, "action_up_fraction_mean"].iloc[0]) for v in variants]
    down = [
        float(final_behavior.loc[final_behavior["variant"] == v, "action_down_fraction_mean"].iloc[0])
        for v in variants
    ]
    noop = [
        float(final_behavior.loc[final_behavior["variant"] == v, "action_noop_fraction_mean"].iloc[0])
        for v in variants
    ]
    axes[1, 1].bar(x - width, up, width, label="up", color="#2A9D8F", alpha=0.9)
    axes[1, 1].bar(x, down, width, label="down", color="#D1495B", alpha=0.85)
    axes[1, 1].bar(x + width, noop, width, label="noop", color="#7A7A7A", alpha=0.85)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([SHORT_LABELS[v] for v in variants])
    axes[1, 1].set_ylabel("Action fraction")
    axes[1, 1].set_title("Final evaluation action mix")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_freeway_behavior.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_scores(phase_summary: pd.DataFrame, figures_dir: Path) -> None:
    if phase_summary.empty:
        return
    variants = ordered_variants(phase_summary)
    phases = list(phase_summary["phase"].dropna().unique())
    x = np.arange(len(phases))
    width = 0.16
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offset_start = -width * (len(variants) - 1) / 2
    for idx, variant in enumerate(variants):
        subset = phase_summary[phase_summary["variant"] == variant].set_index("phase")
        means = [float(subset.loc[phase, "score_mean"]) if phase in subset.index else np.nan for phase in phases]
        errs = [
            0.0
            if phase not in subset.index or pd.isna(subset.loc[phase, "score_ci95"])
            else float(subset.loc[phase, "score_ci95"])
            for phase in phases
        ]
        ax.bar(
            x + offset_start + idx * width,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=SHORT_LABELS[variant].replace("\n", " "),
            color=COLORS[variant],
            alpha=0.9,
        )
    add_freeway_reference_lines(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Mean evaluation score in phase")
    ax.set_title("Early, middle, and late Freeway performance")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_phase_scores.png", bbox_inches="tight")
    plt.close(fig)


def save_figures(
    figures_dir: Path,
    summary_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    paired: pd.DataFrame,
    eval_curve: pd.DataFrame,
    final_behavior: pd.DataFrame,
    phase_summary: pd.DataFrame,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()
    plot_learning_curves(eval_curve, figures_dir)
    plot_final_score(summary_df, figures_dir)
    plot_paired_deltas(paired, figures_dir)
    plot_sample_efficiency(variant_summary, summary_df, figures_dir)
    plot_stability(variant_summary, figures_dir)
    plot_optimization(variant_summary, figures_dir)
    plot_behavior(final_behavior, figures_dir)
    plot_phase_scores(phase_summary, figures_dir)


def performance_table(variant_summary: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in variant_summary.iterrows():
        rows.append(
            [
                row["variant_label"],
                int(row["n"]),
                mean_ci_text(row["final_score_mean"], row["final_score_ci95"], digits=2),
                format_number(row["final_score_std"], 2),
                mean_ci_text(row["best_score_mean"], row["best_score_ci95"], digits=2),
                mean_ci_text(row["normalized_auc_mean"], row["normalized_auc_ci95"], digits=2),
                f"{int(row['final_solved_count'])}/{int(row['n'])}",
                f"{int(row['threshold_reach_count'])}/{int(row['n'])}",
                f"{int(row['score_22_5_reach_count'])}/{int(row['n'])}",
                format_steps(row["first_reach_step_median"]),
                format_number(row["collapse_count_mean"], 1),
                format_number(row["best_final_gap_mean"], 2),
            ]
        )
    return markdown_table(
        [
            "Variant",
            "Seeds",
            "Final score mean +/- CI",
            "Seed std",
            "Best score mean +/- CI",
            "Norm. AUC mean +/- CI",
            "Final >=15",
            "Reached 15",
            "Reached 22.5",
            "Median first >=15",
            "Collapses/run",
            "Best-final gap",
        ],
        rows,
    )


def paired_table(paired: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in paired.iterrows():
        rows.append(
            [
                row["variant_label"],
                mean_ci_text(row["final_delta_mean"], row["final_delta_ci95"], digits=2),
                format_percent(row["final_win_rate_vs_baseline"], 0),
                mean_ci_text(row["normalized_auc_delta_mean"], row["normalized_auc_delta_ci95"], digits=2),
                mean_ci_text(row["first_reach_step_delta_mean"], row["first_reach_step_delta_ci95"], digits=0),
                mean_ci_text(row["collapse_delta_mean"], row["collapse_delta_ci95"], digits=1),
            ]
        )
    return markdown_table(
        [
            "Variant vs DQN",
            "Final score delta",
            "Seed win rate",
            "Norm. AUC delta",
            "First >=15 step delta",
            "Collapse delta",
        ],
        rows,
    )


def seed_table(summary_df: pd.DataFrame) -> str:
    pivot = summary_df.pivot(index="seed", columns="variant", values="final_score")
    rows: list[list[object]] = []
    for seed, row in pivot.sort_index().iterrows():
        values = []
        for variant in VARIANT_ORDER:
            values.append(format_number(row[variant], 2) if variant in pivot.columns else "n/a")
        rows.append([int(seed)] + values)
    return markdown_table(["Seed"] + [SHORT_LABELS[v].replace("\n", " ") for v in VARIANT_ORDER], rows)


def behavior_table(final_behavior: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in final_behavior.iterrows():
        rows.append(
            [
                row["variant_label"],
                mean_ci_text(row["final_score_mean"], row["final_score_ci95"], digits=2),
                format_percent(row["score_15_0_success_rate_mean"], 0),
                format_percent(row["score_22_5_success_rate_mean"], 0),
                format_percent(row["zero_score_rate_mean"], 0),
                format_percent(row["action_up_fraction_mean"], 0),
                format_percent(row["action_down_fraction_mean"], 0),
                format_percent(row["action_noop_fraction_mean"], 0),
            ]
        )
    return markdown_table(
        [
            "Variant",
            "Final score mean +/- CI",
            "Score >=15",
            "Score >=22.5",
            "Zero score",
            "Up action",
            "Down action",
            "Noop action",
        ],
        rows,
    )


def write_metrics_tutorial(analysis_dir: Path) -> None:
    content = """# Freeway Metrics Tutorial

This guide explains the metrics used in the Freeway up-biased DQN analysis. In this run, score and return are the same logged value. A score of 15 is the configured success threshold. The reference scores in the run configuration are random 0.0, human 29.6, and DQN 30.3.

## Evaluation Score

`return_mean` in `eval_metrics.jsonl` is the mean score across the evaluation episodes for one checkpoint. `final_eval_return_mean` in `summary.json` is the last evaluation score. `best_eval_return_mean` is the best checkpoint score reached during training.

`normalized_area_under_eval_curve` summarizes the whole learning curve. Higher values mean the agent scored well earlier, stayed high longer, or both. It is in score units because the raw area is divided by the 1M-step training budget.

## Thresholds and Sample Efficiency

The report tracks first reach steps for score thresholds 5.0, 10.0, 15.0, and 22.5. The main success threshold is 15.0. The first reach step is the first evaluation checkpoint where the mean score reached the threshold.

The sustained threshold step uses the same configured rule as training: the score must stay above the threshold for a window of neighboring evaluation checkpoints.

## Stability

`best_final_gap` is best score minus final score. A large value means the run learned a better policy and then lost part of it by the final checkpoint.

`catastrophic_collapse_count` counts large drops after crossing the success threshold. Read it with the learning curve because a run can have few collapses simply if it spends little time in a high-score regime.

## Optimization and Q Diagnostics

TD loss and TD error describe how hard the Bellman regression problem was near the end of training. They are not policy-quality metrics by themselves. Use them after checking evaluation score.

The Q overestimation proxy compares the online max Q value with the target-side Q value. It is useful for relative diagnostics, but it is affected by Q-value scale and the replay states sampled late in training.

## Freeway Behavior Metrics

`zero_score_rate` is the fraction of evaluation episodes that scored zero. In this result set it should be near zero for trained agents.

`action_up_fraction_mean`, `action_down_fraction_mean`, and `action_noop_fraction_mean` summarize the final policy's action mix during evaluation. Because this experiment used an up-biased exploration setting during training, these action fractions help check whether the final greedy policy still relies heavily on upward movement or also learns when to wait.
"""
    (analysis_dir / "freeway_metrics_tutorial.md").write_text(content, encoding="utf-8")


def write_source_digest(analysis_dir: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    digest_path = analysis_dir / "digest.txt"
    subprocess.run(
        ["gitingest", "rl_analysis", "-i", "*.py", "-o", str(digest_path.resolve())],
        cwd=repo_root,
        check=True,
    )


def write_report(
    analysis_dir: Path,
    summary_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    paired: pd.DataFrame,
    final_behavior: pd.DataFrame,
    report_name: str,
    report_title: str,
    output_root: Path,
    env_slug: str,
) -> None:
    vs = variant_summary.set_index("variant")
    baseline = vs.loc["dqn"]
    double = vs.loc["double_dqn"]
    dueling = vs.loc["dueling_dqn"]
    double_dueling = vs.loc["double_dueling_dqn"]

    paired_index = paired.set_index("variant")
    best_final_variant = variant_summary.sort_values("final_score_mean", ascending=False).iloc[0]
    best_auc_variant = variant_summary.sort_values("normalized_auc_mean", ascending=False).iloc[0]
    most_stable_variant = variant_summary.sort_values(["collapse_count_mean", "best_final_gap_mean"]).iloc[0]
    seed_count = int(variant_summary["n"].max())
    output_path = output_root / env_slug
    success_threshold = summary_df["success_threshold"].dropna().iloc[0]
    up_bias_values = sorted(summary_df["freeway_up_bias"].dropna().unique())
    up_bias_text = ", ".join(format_number(value, 2) for value in up_bias_values) if up_bias_values else "n/a"

    def paired_sentence(variant: str) -> str:
        row = paired_index.loc[variant]
        speed = (
            "no paired threshold comparison"
            if math.isnan(float(row["first_reach_step_delta_mean"]))
            else f"a first-reach delta of {format_signed(row['first_reach_step_delta_mean'], 0)} steps"
        )
        return (
            f"Against DQN on matched seeds, it changed final score by "
            f"{mean_ci_text(row['final_delta_mean'], row['final_delta_ci95'], digits=2)}, "
            f"won {format_percent(row['final_win_rate_vs_baseline'], 0)} of seeds, had "
            f"{mean_ci_text(row['normalized_auc_delta_mean'], row['normalized_auc_delta_ci95'], digits=2)} "
            f"normalized-AUC delta, and had {speed}."
        )

    double_note = (
        "Double DQN had the strongest mean final score in this run."
        if str(best_final_variant["variant"]) == "double_dqn"
        else f"Double DQN did not beat {best_final_variant['variant_label']} on mean final score."
    )
    dueling_note = (
        "Dueling DQN had the strongest whole-training AUC in this run."
        if str(best_auc_variant["variant"]) == "dueling_dqn"
        else f"Dueling DQN trailed {best_auc_variant['variant_label']} on whole-training AUC."
    )
    combined_note = (
        "The combined variant did not clearly inherit the best property of each separate extension."
        if float(double_dueling["final_score_mean"]) < max(float(double["final_score_mean"]), float(dueling["final_score_mean"]))
        else "The combined variant was competitive with the separate extensions."
    )

    report = f"""# {report_title}

## Data and Method

This report analyzes `{output_path}/` with the four requested variants: DQN, Double DQN, Dueling DQN, and Double + Dueling DQN. There are {len(summary_df)} completed runs, with {seed_count} seeds per variant. Each run used {format_steps(summary_df['total_env_steps'].max())} environment steps, about {format_steps(summary_df['total_ale_frames'].max())} ALE frames, and Freeway up-bias `{up_bias_text}` during exploration. The success threshold is score >= {format_number(success_threshold, 1)}.

All statistics are computed from `summary.json`, `eval_metrics.jsonl`, and `train_update_metrics.jsonl`. Intervals are 95 percent t-intervals across seeds. With {seed_count} seeds per variant, these intervals are uncertainty estimates rather than strong significance claims.

Figures:

![Evaluation learning curves](figures/freeway/fig_eval_learning_curves.png)

![Final score distribution](figures/freeway/fig_final_score_distribution.png)

## Overall Performance Summary

{performance_table(variant_summary)}

The strongest final mean score is from {best_final_variant['variant_label']} at {mean_ci_text(best_final_variant['final_score_mean'], best_final_variant['final_score_ci95'], digits=2)}. The strongest whole-training average is from {best_auc_variant['variant_label']} at {mean_ci_text(best_auc_variant['normalized_auc_mean'], best_auc_variant['normalized_auc_ci95'], digits=2)} normalized AUC. The most stable variant by collapse count and best-final gap is {most_stable_variant['variant_label']}.

Per-seed final scores:

{seed_table(summary_df)}

Paired comparison against DQN:

{paired_table(paired)}

![Paired final score deltas](figures/freeway/fig_paired_final_delta_vs_dqn.png)

## Baseline DQN

{VARIANT_NOTES['dqn']} It finished with mean score {mean_ci_text(baseline['final_score_mean'], baseline['final_score_ci95'], digits=2)} and reached score 15 in {count_text(int(baseline['threshold_reach_count']), int(baseline['n']))}. It also reached score 22.5 in {count_text(int(baseline['score_22_5_reach_count']), int(baseline['n']))}. Its normalized AUC was {mean_ci_text(baseline['normalized_auc_mean'], baseline['normalized_auc_ci95'], digits=2)}.

The baseline is a strong reference for this up-biased setting. Its final score is close to the human reference score of {format_number(REFERENCE_HUMAN, 1)}, but its average best-final gap of {format_number(baseline['best_final_gap_mean'], 2)} shows some late loss from the best checkpoint.

![Sample efficiency and AUC](figures/freeway/fig_sample_efficiency_auc.png)

![Training phase scores](figures/freeway/fig_phase_scores.png)

## Double DQN

{VARIANT_NOTES['double_dqn']} It finished with mean score {mean_ci_text(double['final_score_mean'], double['final_score_ci95'], digits=2)}, reached score 15 in {count_text(int(double['threshold_reach_count']), int(double['n']))}, and reached score 22.5 in {count_text(int(double['score_22_5_reach_count']), int(double['n']))}.

{paired_sentence('double_dqn')} {double_note} Its Q-overestimation proxy was {format_number(double['q_overestimation_proxy_last10_mean'], 4)}, compared with {format_number(baseline['q_overestimation_proxy_last10_mean'], 4)} for DQN, so the score result should be read together with the full learning curve rather than from that proxy alone.

## Dueling DQN

{VARIANT_NOTES['dueling_dqn']} It finished with mean score {mean_ci_text(dueling['final_score_mean'], dueling['final_score_ci95'], digits=2)}, reached score 15 in {count_text(int(dueling['threshold_reach_count']), int(dueling['n']))}, and reached score 22.5 in {count_text(int(dueling['score_22_5_reach_count']), int(dueling['n']))}.

{paired_sentence('dueling_dqn')} {dueling_note} Its average collapse count was {format_number(dueling['collapse_count_mean'], 1)}, compared with {format_number(baseline['collapse_count_mean'], 1)} for DQN.

## Double + Dueling DQN

{VARIANT_NOTES['double_dueling_dqn']} It finished with mean score {mean_ci_text(double_dueling['final_score_mean'], double_dueling['final_score_ci95'], digits=2)}, reached score 15 in {count_text(int(double_dueling['threshold_reach_count']), int(double_dueling['n']))}, and reached score 22.5 in {count_text(int(double_dueling['score_22_5_reach_count']), int(double_dueling['n']))}.

{paired_sentence('double_dueling_dqn')} {combined_note}

![Stability and regression diagnostics](figures/freeway/fig_stability_regression.png)

## Optimization, Q-Value, and Behavior Diagnostics

The optimization diagnostics are best used as explanations after checking evaluation score. Low TD loss does not automatically mean a better Freeway policy, and Q scale can differ across variants.

![Optimization and Q diagnostics](figures/freeway/fig_optimization_q_diagnostics.png)

Final evaluation behavior summary:

{behavior_table(final_behavior)}

The behavior table and figure show that all variants avoid zero-score episodes at the end, so the main comparison is score quality and action mix rather than basic task discovery.

![Freeway behavior diagnostics](figures/freeway/fig_freeway_behavior.png)

## Main Conclusions

All four requested variants learned useful Freeway policies under the up-biased exploration setting. Every seed reached the configured score-15 threshold, so final quality, whole-training AUC, and late stability are more informative than simple success rate.

{best_final_variant['variant_label']} had the best final mean score. {best_auc_variant['variant_label']} had the best normalized AUC, which means it had the strongest score profile over the full 1M-step budget.

The combined Double + Dueling variant was not automatically better than the separate extensions in this run. The result supports comparing these variants by matched seeds and learning curves, not only by their architectural intent.
"""
    (analysis_dir / report_name).write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir)
    figures_dir = analysis_dir / "figures" / "freeway"
    tables_dir = analysis_dir / "tables"

    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_df, eval_df, _update_df = load_data(output_root, args.env_slug)
    variant_summary = build_variant_summary(summary_df)
    paired = build_paired_comparison(summary_df)
    eval_curve = build_eval_curve(eval_df)
    final_behavior = build_final_behavior(eval_df)
    phase_summary = build_phase_summary(eval_df)

    save_tables(tables_dir, summary_df, variant_summary, paired, eval_curve, final_behavior, phase_summary)
    save_figures(figures_dir, summary_df, variant_summary, paired, eval_curve, final_behavior, phase_summary)
    write_metrics_tutorial(analysis_dir)
    write_source_digest(analysis_dir)
    write_report(
        analysis_dir,
        summary_df,
        variant_summary,
        paired,
        final_behavior,
        args.report_name,
        args.report_title,
        output_root,
        args.env_slug,
    )

    print(f"Wrote report to {analysis_dir / args.report_name}")
    print(f"Wrote metrics tutorial to {analysis_dir / 'freeway_metrics_tutorial.md'}")
    print(f"Wrote source digest to {analysis_dir / 'digest.txt'}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote tables to {tables_dir}")


if __name__ == "__main__":
    main()
