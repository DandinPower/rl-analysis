#!/usr/bin/env python3
"""Generate LunarLander DQN analysis tables, figures, and markdown reports."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/private/tmp")) / "rl-analysis-matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_ORDER = [
    "dqn",
    "dqn_no_target",
    "dqn_no_replay",
    "double_dqn",
    "dueling_dqn",
    "double_dueling_dqn",
]

VARIANT_LABELS = {
    "dqn": "DQN baseline",
    "dqn_no_target": "DQN without target network",
    "dqn_no_replay": "DQN without replay buffer",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "double_dueling_dqn": "Double + Dueling DQN",
}

SHORT_LABELS = {
    "dqn": "DQN",
    "dqn_no_target": "No target",
    "dqn_no_replay": "No replay",
    "double_dqn": "Double",
    "dueling_dqn": "Dueling",
    "double_dueling_dqn": "Double+\nDueling",
}

VARIANT_NOTES = {
    "dqn": "Baseline with experience replay and a periodically copied target network.",
    "dqn_no_target": "Removes the target network, so the bootstrap target moves with the online network.",
    "dqn_no_replay": "Removes replay, so updates use recent correlated transitions with little sample reuse.",
    "double_dqn": "Uses the online network to select the next action and the target network to evaluate it.",
    "dueling_dqn": "Splits the network head into state-value and action-advantage streams.",
    "double_dueling_dqn": "Combines Double DQN target selection with the dueling value/advantage architecture.",
}

COLORS = {
    "dqn": "#2E86AB",
    "dqn_no_target": "#D1495B",
    "dqn_no_replay": "#7A7A7A",
    "double_dqn": "#2A9D8F",
    "dueling_dqn": "#F4A261",
    "double_dueling_dqn": "#6D597A",
}

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
    parser.add_argument("--output-root", default="output", help="Root containing output/lunarlander_v3.")
    parser.add_argument("--analysis-dir", default="analysis", help="Directory for generated analysis artifacts.")
    parser.add_argument("--env-slug", default="lunarlander_v3", help="Environment slug under output-root.")
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

            row = {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": int(summary["seed"]),
                "run_id": summary.get("run_id", run_dir.name),
                "run_dir": str(run_dir),
                "training_completed": bool(summary.get("training_completed", False)),
                "total_env_steps": to_float(summary.get("total_env_steps")),
                "total_updates": to_float(summary.get("total_updates")),
                "total_episodes": to_float(summary.get("total_episodes")),
                "wall_time_total_sec": to_float(compute.get("wall_time_total_sec")),
                "env_steps_per_second": to_float(compute.get("env_steps_per_second")),
                "updates_per_second": to_float(compute.get("updates_per_second")),
                "final_return": to_float(performance.get("final_eval_return_mean")),
                "final_return_median": to_float(performance.get("final_eval_return_median")),
                "final_return_std_within_eval": to_float(performance.get("final_eval_return_std")),
                "best_return": to_float(performance.get("best_eval_return_mean")),
                "best_eval_step": to_float(performance.get("best_eval_step")),
                "area_under_eval_curve": to_float(performance.get("area_under_eval_curve")),
                "normalized_auc": to_float(performance.get("normalized_area_under_eval_curve")),
                "final_train_return_mean_last_100": to_float(performance.get("final_train_return_mean_last_100")),
                "final_train_return_std_last_100": to_float(performance.get("final_train_return_std_last_100")),
                "success_threshold": to_float(sample_efficiency.get("threshold")),
                "first_reach_step": to_float(sample_efficiency.get("first_step_reaching_threshold")),
                "first_sustained_step": to_float(sample_efficiency.get("first_step_sustained_threshold")),
                "reached_threshold": not bool(sample_efficiency.get("never_reached_threshold", True)),
                "collapse_count": to_float(stability.get("catastrophic_collapse_count")),
                "largest_eval_drop": to_float(stability.get("largest_eval_drop")),
                "largest_eval_drop_fraction": to_float(stability.get("largest_eval_drop_fraction")),
                "eval_return_std_across_time": to_float(stability.get("eval_return_std_across_time")),
                "rolling_return_std_mean": to_float(stability.get("rolling_return_std_mean")),
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
                "use_target_network": bool(get_nested(config, "algorithm.use_target_network", False)),
                "use_replay_buffer": bool(get_nested(config, "algorithm.use_replay_buffer", False)),
                "use_double_dqn": bool(get_nested(config, "algorithm.use_double_dqn", False)),
                "use_dueling_network": bool(get_nested(config, "algorithm.use_dueling_network", False)),
                "learning_rate": to_float(get_nested(config, "optimizer.learning_rate")),
                "gamma": to_float(get_nested(config, "dqn_hyperparameters.gamma")),
                "batch_size": to_float(get_nested(config, "dqn_hyperparameters.batch_size")),
                "target_update_frequency": to_float(
                    get_nested(config, "dqn_hyperparameters.target_update_frequency_env_steps")
                ),
                "epsilon_decay_steps": to_float(get_nested(config, "exploration.epsilon_decay_env_steps")),
            }
            row["best_final_gap"] = row["best_return"] - row["final_return"]
            row["final_solved"] = bool(row["final_return"] >= row["success_threshold"])
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
            "final_return_mean": safe_mean(group["final_return"]),
            "final_return_std": safe_std(group["final_return"]),
            "final_return_ci95": ci95(group["final_return"]),
            "final_return_median": float(group["final_return"].median()),
            "final_return_min": float(group["final_return"].min()),
            "final_return_max": float(group["final_return"].max()),
            "final_solved_rate": safe_mean(solved),
            "final_solved_count": int(group["final_solved"].sum()),
            "best_return_mean": safe_mean(group["best_return"]),
            "best_return_std": safe_std(group["best_return"]),
            "best_return_ci95": ci95(group["best_return"]),
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
            "eval_return_std_across_time_mean": safe_mean(group["eval_return_std_across_time"]),
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
        records.append(record)
    return pd.DataFrame(records)


def build_eval_curve(eval_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    for (variant, step), group in eval_df.groupby(["variant", "global_env_step"], observed=True):
        if pd.isna(variant):
            continue
        records.append(
            {
                "variant": str(variant),
                "variant_label": VARIANT_LABELS[str(variant)],
                "global_env_step": int(step),
                "n": int(len(group)),
                "return_mean": safe_mean(group["return_mean"]),
                "return_std": safe_std(group["return_mean"]),
                "return_ci95": ci95(group["return_mean"]),
                "success_rate_mean": safe_mean(group.get("lunarlander_eval.success_rate", pd.Series(dtype=float))),
                "landing_success_rate_mean": safe_mean(
                    group.get("lunarlander_eval.landing_success_rate", pd.Series(dtype=float))
                ),
                "crash_rate_mean": safe_mean(group.get("lunarlander_eval.crash_rate", pd.Series(dtype=float))),
                "timeout_rate_mean": safe_mean(group.get("lunarlander_eval.timeout_rate", pd.Series(dtype=float))),
                "main_engine_fraction_mean": safe_mean(
                    group.get("lunarlander_eval.main_engine_action_fraction_mean", pd.Series(dtype=float))
                ),
                "side_engine_fraction_mean": safe_mean(
                    group.get("lunarlander_eval.side_engine_action_fraction_mean", pd.Series(dtype=float))
                ),
            }
        )
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
        "final_eval_return_mean": "return_mean",
        "final_eval_success_rate": "lunarlander_eval.success_rate",
        "landing_success_rate": "lunarlander_eval.landing_success_rate",
        "crash_rate": "lunarlander_eval.crash_rate",
        "timeout_rate": "lunarlander_eval.timeout_rate",
        "both_legs_contact_rate": "lunarlander_eval.both_legs_contact_rate",
        "one_leg_contact_rate": "lunarlander_eval.one_leg_contact_rate",
        "no_leg_contact_rate": "lunarlander_eval.no_leg_contact_rate",
        "main_engine_fraction": "lunarlander_eval.main_engine_action_fraction_mean",
        "side_engine_fraction": "lunarlander_eval.side_engine_action_fraction_mean",
        "episode_length_mean": "episode_length_mean",
        "action_entropy_mean": "action_entropy_mean",
    }
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
            phases["global_env_step"] <= 100_000,
            (phases["global_env_step"] > 100_000) & (phases["global_env_step"] <= 300_000),
            phases["global_env_step"] > 300_000,
        ],
        ["early: 10k-100k", "middle: 110k-300k", "late: 310k-500k"],
        default="unknown",
    )
    per_seed = (
        phases.groupby(["variant", "seed", "phase"], observed=True)["return_mean"]
        .mean()
        .reset_index(name="phase_return_mean")
    )
    records: list[dict] = []
    for (variant, phase), group in per_seed.groupby(["variant", "phase"], observed=True):
        records.append(
            {
                "variant": str(variant),
                "variant_label": VARIANT_LABELS[str(variant)],
                "phase": str(phase),
                "n": int(len(group)),
                "return_mean": safe_mean(group["phase_return_mean"]),
                "return_std": safe_std(group["phase_return_mean"]),
                "return_ci95": ci95(group["phase_return_mean"]),
            }
        )
    phase_df = pd.DataFrame(records)
    phase_order = ["early: 10k-100k", "middle: 110k-300k", "late: 310k-500k"]
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
        final_delta = o["final_return"].to_numpy(dtype=float) - b["final_return"].to_numpy(dtype=float)
        best_delta = o["best_return"].to_numpy(dtype=float) - b["best_return"].to_numpy(dtype=float)
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
    summary_df.to_csv(tables_dir / "lunarlander_run_summary.csv", index=False)
    variant_summary.to_csv(tables_dir / "lunarlander_variant_summary.csv", index=False)
    paired.to_csv(tables_dir / "lunarlander_paired_vs_dqn.csv", index=False)
    eval_curve.to_csv(tables_dir / "lunarlander_eval_curve_summary.csv", index=False)
    final_behavior.to_csv(tables_dir / "lunarlander_final_behavior_summary.csv", index=False)
    phase_summary.to_csv(tables_dir / "lunarlander_phase_return_summary.csv", index=False)


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


def ordered_values(df: pd.DataFrame, column: str) -> list[float]:
    return [float(df.loc[df["variant"] == variant, column].iloc[0]) for variant in VARIANT_ORDER if variant in set(df["variant"])]


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


def plot_learning_curves(eval_curve: pd.DataFrame, figures_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for variant in ordered_variants(eval_curve):
        curve = eval_curve[eval_curve["variant"] == variant]
        x = curve["global_env_step"].to_numpy(dtype=float)
        y = curve["return_mean"].to_numpy(dtype=float)
        ci = curve["return_ci95"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, label=SHORT_LABELS[variant].replace("\n", " "), color=COLORS[variant], linewidth=2)
        ax.fill_between(x, y - ci, y + ci, color=COLORS[variant], alpha=0.12, linewidth=0)
    ax.axhline(200, color="#333333", linewidth=1.1, linestyle="--", label="success threshold")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Evaluation return mean")
    ax.set_title("Evaluation learning curves across seeds")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_eval_learning_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_final_return(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    variants = ordered_variants(summary_df)
    data = [summary_df[summary_df["variant"] == variant]["final_return"].to_numpy(dtype=float) for variant in variants]
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
    ax.axhline(200, color="#333333", linestyle="--", linewidth=1.1, label="success threshold")
    ax.set_ylabel("Final evaluation return")
    ax.set_title("Final return distribution across five seeds")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_final_return_distribution.png", bbox_inches="tight")
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
    ax.set_xlabel("Final return delta vs baseline, same seed")
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
        "Average eval return over training",
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
    axes[1].set_ylabel("Median first step reaching return >= 200")
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
        "Best return minus final return",
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
        "final_eval_success_rate_mean",
        "final_eval_success_rate_ci95",
        "Rate (%)",
        "Success rate",
        percent=True,
    )
    bar_with_ci(
        axes[0, 1],
        final_behavior,
        "crash_rate_mean",
        "crash_rate_ci95",
        "Rate (%)",
        "Crash rate",
        percent=True,
    )
    bar_with_ci(
        axes[1, 0],
        final_behavior,
        "timeout_rate_mean",
        "timeout_rate_ci95",
        "Rate (%)",
        "Timeout rate",
        percent=True,
    )

    variants = ordered_variants(final_behavior)
    x = np.arange(len(variants))
    width = 0.38
    main = [float(final_behavior.loc[final_behavior["variant"] == v, "main_engine_fraction_mean"].iloc[0]) for v in variants]
    side = [float(final_behavior.loc[final_behavior["variant"] == v, "side_engine_fraction_mean"].iloc[0]) for v in variants]
    axes[1, 1].bar(x - width / 2, main, width, label="main engine", color="#2E86AB", alpha=0.9)
    axes[1, 1].bar(x + width / 2, side, width, label="side engines", color="#D1495B", alpha=0.85)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([SHORT_LABELS[v] for v in variants])
    axes[1, 1].set_ylabel("Action fraction")
    axes[1, 1].set_title("Engine use")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_lunarlander_behavior.png", bbox_inches="tight")
    plt.close(fig)


def plot_phase_returns(phase_summary: pd.DataFrame, figures_dir: Path) -> None:
    if phase_summary.empty:
        return
    variants = ordered_variants(phase_summary)
    phases = list(phase_summary["phase"].dropna().unique())
    x = np.arange(len(phases))
    width = 0.12
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offset_start = -width * (len(variants) - 1) / 2
    for idx, variant in enumerate(variants):
        subset = phase_summary[phase_summary["variant"] == variant].set_index("phase")
        means = [float(subset.loc[phase, "return_mean"]) if phase in subset.index else np.nan for phase in phases]
        errs = [
            0.0 if phase not in subset.index or pd.isna(subset.loc[phase, "return_ci95"]) else float(subset.loc[phase, "return_ci95"])
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
    ax.axhline(200, color="#333333", linewidth=1.1, linestyle="--", label="success threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("Mean evaluation return in phase")
    ax.set_title("Early, middle, and late training performance")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_phase_returns.png", bbox_inches="tight")
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
    plot_final_return(summary_df, figures_dir)
    plot_paired_deltas(paired, figures_dir)
    plot_sample_efficiency(variant_summary, summary_df, figures_dir)
    plot_stability(variant_summary, figures_dir)
    plot_optimization(variant_summary, figures_dir)
    plot_behavior(final_behavior, figures_dir)
    plot_phase_returns(phase_summary, figures_dir)


def performance_table(variant_summary: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in variant_summary.iterrows():
        rows.append(
            [
                row["variant_label"],
                int(row["n"]),
                mean_ci_text(row["final_return_mean"], row["final_return_ci95"]),
                format_number(row["final_return_std"], 1),
                mean_ci_text(row["best_return_mean"], row["best_return_ci95"]),
                mean_ci_text(row["normalized_auc_mean"], row["normalized_auc_ci95"]),
                f"{int(row['threshold_reach_count'])}/{int(row['n'])}",
                format_steps(row["first_reach_step_median"]),
                format_number(row["collapse_count_mean"], 1),
                format_number(row["best_final_gap_mean"], 1),
            ]
        )
    return markdown_table(
        [
            "Variant",
            "Seeds",
            "Final return mean +/- CI",
            "Seed std",
            "Best return mean +/- CI",
            "Norm. AUC mean +/- CI",
            "Reached 200",
            "Median first reach",
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
                mean_ci_text(row["final_delta_mean"], row["final_delta_ci95"]),
                format_percent(row["final_win_rate_vs_baseline"], 0),
                mean_ci_text(row["normalized_auc_delta_mean"], row["normalized_auc_delta_ci95"]),
                mean_ci_text(row["first_reach_step_delta_mean"], row["first_reach_step_delta_ci95"], digits=0),
                mean_ci_text(row["collapse_delta_mean"], row["collapse_delta_ci95"]),
            ]
        )
    return markdown_table(
        [
            "Variant vs DQN",
            "Final return delta",
            "Seed win rate",
            "Norm. AUC delta",
            "First-reach step delta",
            "Collapse delta",
        ],
        rows,
    )


def seed_table(summary_df: pd.DataFrame) -> str:
    pivot = summary_df.pivot(index="seed", columns="variant", values="final_return")
    rows: list[list[object]] = []
    for seed, row in pivot.sort_index().iterrows():
        rows.append([int(seed)] + [format_number(row.get(variant, np.nan), 1) for variant in VARIANT_ORDER])
    return markdown_table(["Seed"] + [SHORT_LABELS[v].replace("\n", " ") for v in VARIANT_ORDER], rows)


def behavior_table(final_behavior: pd.DataFrame) -> str:
    rows: list[list[object]] = []
    for _, row in final_behavior.iterrows():
        rows.append(
            [
                row["variant_label"],
                format_percent(row["final_eval_success_rate_mean"], 0),
                format_percent(row["landing_success_rate_mean"], 0),
                format_percent(row["crash_rate_mean"], 0),
                format_percent(row["timeout_rate_mean"], 0),
                format_percent(row["main_engine_fraction_mean"], 0),
                format_percent(row["side_engine_fraction_mean"], 0),
            ]
        )
    return markdown_table(
        ["Variant", "Success", "Landing success", "Crash", "Timeout", "Main engine", "Side engines"],
        rows,
    )


def write_metrics_tutorial(analysis_dir: Path) -> None:
    content = """# Metrics Tutorial for LunarLander DQN Experiments

This guide explains how to read the result logs in this repository and what to compare when analyzing DQN-style reinforcement learning runs. It assumes you already understand normal ML training ideas such as optimization loss, validation curves, overfitting, randomness, and ablation studies. The main difference is that in RL the model changes the data it will see next. A better policy visits different states, gets different rewards, and creates a different training distribution.

## 1. Start With the Question, Not the Loss

For supervised learning, a lower validation loss is often close to the final goal. For DQN, the TD loss is only the Bellman regression loss on sampled transitions. A low TD loss can mean the value function is accurate, but it can also mean the replay data is easy, narrow, or unhelpful. The first question should always be: does the policy actually get better return in evaluation?

In these logs, the main policy-quality fields are in `eval_metrics.jsonl` and `summary.json`.

`return_mean` is the average evaluation return over a fixed number of evaluation episodes. This is the closest metric to final task performance. For LunarLander, a return above 200 is usually treated as solved.

`return_std`, `return_p25`, `return_p75`, `return_min`, and `return_max` show how much one evaluation batch varies. High within-evaluation variance means the same checkpoint can sometimes land well and sometimes fail.

`final_eval_return_mean` in `summary.json` is the last evaluation return. It answers: how good was the policy at the end of training?

`best_eval_return_mean` answers: did the run ever find a good policy? This matters because DQN can learn and later regress.

`best_final_gap`, computed as best return minus final return, answers: did the run keep the good policy or lose it later?

## 2. Use Learning Curves to Separate Speed From Final Quality

A single final score hides the training path. In DQN, two runs can have the same final return but very different stories. One may learn early and stay stable. Another may fail for most of training and recover near the end. The evaluation learning curve shows this.

Check `global_env_step` against `return_mean`. Then compare variants at the same environment step, not at the same wall-clock time. Environment steps measure sample budget and make the ablation fair.

Useful questions:

Does the curve cross 200 early, late, or never?

Does it stay above 200 after crossing, or does it collapse?

Does it improve smoothly, or jump sharply after a long flat period?

Does the mean curve hide large seed variance?

`area_under_eval_curve` and `normalized_area_under_eval_curve` summarize the whole learning curve. In this report, normalized AUC is roughly the average evaluation return over the training budget. A high normalized AUC means the agent was useful for more of training, not only at the end.

## 3. Sample Efficiency: How Much Experience Was Needed?

Sample efficiency is about how many environment interactions were needed before the agent became good. In these logs, the main fields are:

`first_step_reaching_threshold`: the first evaluation step where mean return reached the success threshold.

`first_step_sustained_threshold`: the first step where the run stayed above threshold for a window of evaluations.

`never_reached_threshold`: whether the run failed to reach the threshold at all.

For a fair comparison, use the same threshold for every variant. For LunarLander, this project uses return >= 200. Compare the median first reach step across seeds, but also report how many seeds reached it. A variant with a fast median over only two successful seeds is not better than a variant that reaches the threshold in all seeds.

## 4. Stability: RL Can Learn and Then Forget

DQN is bootstrapped: it learns Q-values from targets that depend on other Q-values. This can create feedback loops. A policy can become good and then get worse as the value estimates shift.

The main stability fields are:

`catastrophic_collapse_count`: how many times evaluation return dropped by a large fraction after the run had already crossed the success threshold.

`largest_eval_drop`: the largest absolute evaluation drop after a previous best.

`eval_return_std_across_time`: how much the evaluation return moved over training.

`rolling_return_std_mean`: how noisy the recent training returns were.

Always interpret collapse count with threshold reach rate. A run that never learns may have zero collapses only because it never reached a level from which it could collapse.

## 5. Seeds Matter More Than in Many Supervised Runs

RL training is sensitive to random seeds because the seed affects initialization, exploration actions, replay contents, and sometimes environment transitions. Do not trust one seed.

For each variant, compare:

Mean final return across seeds.

Seed standard deviation.

95 percent confidence interval of the seed mean.

How many seeds solved the task at the end.

How many seeds ever reached the threshold.

The per-seed table is also important. If one seed fails completely while four seeds are excellent, the mean may look acceptable but the method is unreliable.

When the same seed IDs are shared across variants, paired seed comparisons are useful. For seed 0, compare every variant against baseline seed 0; for seed 1, compare against baseline seed 1; and so on. This reduces noise from lucky or unlucky seeds.

## 6. Replay Buffer Diagnostics

Experience replay is one of the central DQN stabilizers. It breaks short-term correlation and allows each transition to be reused for multiple updates.

Important fields:

`replay_enabled`: whether replay was used.

`final_buffer_size`: how many transitions were in the buffer at the end.

`sample_age_mean_last_10pct`: how old sampled transitions were late in training. A large sample age means updates used a mix of older and newer experience.

`sample_consecutive_transition_fraction_last_10pct`: how often sampled transitions were adjacent in the original trajectory. Lower values mean better decorrelation.

If replay is removed, the agent usually trains on recent correlated data. That can make learning more like chasing a moving target from a narrow stream of experience. In the report, the no-replay ablation should be read mostly as a test of decorrelation and sample reuse.

## 7. Target Network Diagnostics

DQN uses a target network because the Bellman target contains a neural network prediction. If the same network is used for both prediction and target, the target moves every update. This can make bootstrapping unstable.

Important fields:

`target_network_enabled`: whether a separate target network was used.

`target_update_count`: how many hard or soft target updates occurred.

`online_target_param_l2_mean`: how far the online and target networks were from each other on average.

`target_q_mean_last_10pct`: the mean target-side Q-value late in training.

Removing the target network can sometimes still solve a simple environment, but the expected risk is more oscillation, larger collapses, or worse retention.

## 8. TD Error, TD Loss, and Gradient Metrics

TD error is the difference between the current Q estimate and the Bellman target. It is the DQN analogue of a regression residual, but the target itself is learned and non-stationary.

Important fields:

`td_loss_mean_last_10pct`: average TD loss late in training.

`td_error_abs_mean_last_10pct`: mean absolute TD error late in training.

`td_error_abs_p95_last_10pct`: tail TD error. This is often more useful than the mean because rare bad targets can destabilize training.

`grad_norm_mean_last_10pct` and `grad_norm_max`: gradient scale.

`gradient_clip_fraction`: fraction of updates where clipping was active.

High clipping is not automatically bad. It may mean the optimizer is being protected from large TD errors. But if clipping is high and returns are poor, it suggests the update targets are noisy or unstable.

## 9. Q-Value Diagnostics and Overestimation

DQN can overestimate action values because the max over noisy Q estimates tends to select overestimated actions. Double DQN tries to reduce this by using one network to select the action and the other network to evaluate it.

Important fields:

`online_q_mean_last_10pct`: average online Q-value late in training.

`online_q_max_mean_last_10pct`: average max action value late in training.

`target_q_mean_last_10pct`: average target Q-value late in training.

`q_overestimation_proxy_last10`: a proxy comparing online max values to target values.

`max_q_value_seen`: the largest Q-value observed during training.

Interpret Q metrics together with return. A higher Q scale is not better by itself. If Q-values rise while returns do not, the value function may be overconfident. If Double DQN improves return but the proxy does not decrease, that means this proxy is not capturing the whole effect, or the variant reached a different value scale.

## 10. Dueling Network Diagnostics

Dueling DQN separates Q(s, a) into a state-value stream and an action-advantage stream. This helps when many actions have similar value in a state, because the network can learn that the state is good or bad without having to relearn that fact independently for each action.

Useful logged fields in update metrics include:

`dueling.value_stream_mean` and `dueling.value_stream_std`.

`dueling.advantage_abs_mean`.

`dueling.action_gap_mean`.

The action gap is the difference between the best action value and the next alternatives. A larger useful gap can make the greedy policy more decisive, but a too-large gap with bad return can mean the network is confidently wrong.

## 11. LunarLander-Specific Behavior Metrics

Return tells you whether the agent did well. Environment-specific metrics help explain how.

Useful fields under `lunarlander_eval`:

`success_rate`: fraction of evaluation episodes with return >= 200.

`landing_success_rate`: fraction of episodes ending in a successful landing condition.

`crash_rate`: fraction of episodes ending in a crash.

`timeout_rate`: fraction of episodes that reached the time limit.

`both_legs_contact_rate`, `one_leg_contact_rate`, and `no_leg_contact_rate`: final contact behavior.

`main_engine_action_fraction_mean` and `side_engine_action_fraction_mean`: engine usage. High main-engine use may indicate hovering or inefficient control. High side-engine use may indicate strong attitude correction.

`episode_length_mean`: long episodes can mean stable hovering, slow landing, or timeouts. Use it with success, crash, and timeout rates.

## 12. A Practical Comparison Checklist

For every variant, check these in order.

First, final evaluation return and final success rate. This answers whether the final policy is useful.

Second, best return and best-final gap. This answers whether the run ever learned and whether it retained the learned behavior.

Third, normalized AUC and first reach step. This answers whether learning was sample-efficient.

Fourth, seed standard deviation and paired seed deltas. This answers whether the result is reliable or seed-dependent.

Fifth, collapse count and largest drop. This answers whether the method is stable after success.

Sixth, TD loss, TD error tail, gradient clipping, and Q-value scale. This helps explain training dynamics.

Seventh, LunarLander behavior metrics. This turns scores into a behavioral story: landing, crashing, hovering, timing out, or wasting fuel.

The figures generated by `analysis/analyze_lunarlander.py` follow this checklist: learning curves, final return distributions, sample efficiency, stability, optimization and Q diagnostics, and LunarLander behavior.
"""
    (analysis_dir / "metrics_tutorials.md").write_text(content, encoding="utf-8")


def write_report(
    analysis_dir: Path,
    summary_df: pd.DataFrame,
    variant_summary: pd.DataFrame,
    paired: pd.DataFrame,
    final_behavior: pd.DataFrame,
) -> None:
    vs = variant_summary.set_index("variant")
    baseline = vs.loc["dqn"]
    no_target = vs.loc["dqn_no_target"]
    no_replay = vs.loc["dqn_no_replay"]
    double = vs.loc["double_dqn"]
    dueling = vs.loc["dueling_dqn"]
    double_dueling = vs.loc["double_dueling_dqn"]

    paired_index = paired.set_index("variant")
    dqn_seed_values = summary_df[summary_df["variant"] == "dqn"].sort_values("seed")
    dqn_bad_seeds = dqn_seed_values[dqn_seed_values["final_return"] < dqn_seed_values["success_threshold"]]
    best_final_variant = variant_summary.sort_values("final_return_mean", ascending=False).iloc[0]
    best_auc_variant = variant_summary.sort_values("normalized_auc_mean", ascending=False).iloc[0]

    def paired_sentence(variant: str) -> str:
        row = paired_index.loc[variant]
        speed = (
            "no paired threshold comparison because the variant did not reach the threshold"
            if math.isnan(float(row["first_reach_step_delta_mean"]))
            else f"a first-reach delta of {format_signed(row['first_reach_step_delta_mean'], 0)} steps"
        )
        return (
            f"Against baseline on matched seeds, it changed final return by "
            f"{mean_ci_text(row['final_delta_mean'], row['final_delta_ci95'])}, "
            f"won {format_percent(row['final_win_rate_vs_baseline'], 0)} of seeds, had "
            f"{mean_ci_text(row['normalized_auc_delta_mean'], row['normalized_auc_delta_ci95'])} normalized-AUC delta, "
            f"and had {speed}."
        )

    report = f"""# LunarLander Report v0

## Data and Method

This report analyzes the completed `output/lunarlander_v3/` runs. There are {len(summary_df)} runs: six DQN variants with {int(baseline['n'])} seeds each. Every run used a budget of {format_steps(summary_df['total_env_steps'].max())} environment steps, and the success threshold is return >= {format_number(summary_df['success_threshold'].dropna().iloc[0], 0)}. The analysis uses the latest completed run under each variant/seed directory.

The statistics below are computed with numpy and pandas from `summary.json`, `eval_metrics.jsonl`, and the update logs. Intervals are 95 percent t-intervals across seeds. With only five seeds, the intervals should be read as uncertainty estimates, not as formal proof.

Figures:

![Evaluation learning curves](figures/lunarlander/fig_eval_learning_curves.png)

![Final return distribution](figures/lunarlander/fig_final_return_distribution.png)

## Overall Performance Summary

{performance_table(variant_summary)}

The strongest final mean return is from {best_final_variant['variant_label']} at {mean_ci_text(best_final_variant['final_return_mean'], best_final_variant['final_return_ci95'])}. The strongest whole-training average is from {best_auc_variant['variant_label']} at {mean_ci_text(best_auc_variant['normalized_auc_mean'], best_auc_variant['normalized_auc_ci95'])} normalized AUC. This difference is important: final score rewards the last checkpoint, while normalized AUC rewards being good for more of training.

Per-seed final returns:

{seed_table(summary_df)}

Paired comparison against baseline:

{paired_table(paired)}

![Paired final return deltas](figures/lunarlander/fig_paired_final_delta_vs_dqn.png)

## Baseline DQN Performance Study

The baseline DQN solved the task at least once in all {int(baseline['n'])} seeds. Its final mean return was {mean_ci_text(baseline['final_return_mean'], baseline['final_return_ci95'])}, with a large seed standard deviation of {format_number(baseline['final_return_std'], 1)}. Its best mean return was {mean_ci_text(baseline['best_return_mean'], baseline['best_return_ci95'])}, and its median first step reaching the success threshold was {format_steps(baseline['first_reach_step_median'])}.

The baseline's main strength is early sample efficiency. Its normalized AUC was {mean_ci_text(baseline['normalized_auc_mean'], baseline['normalized_auc_ci95'])}, the highest among these variants. The learning-curve and phase-return figures show that baseline DQN became useful earlier than most extensions.

The weakness is retention. The average best-final gap was {format_number(baseline['best_final_gap_mean'], 1)}, and the average collapse count was {format_number(baseline['collapse_count_mean'], 1)} per run. Two baseline seeds finished below the success threshold: {', '.join(str(int(seed)) for seed in dqn_bad_seeds['seed'])}. This means the baseline often found a good policy, but the final checkpoint was not always the best policy.

![Sample efficiency and AUC](figures/lunarlander/fig_sample_efficiency_auc.png)

![Training phase returns](figures/lunarlander/fig_phase_returns.png)

## Ablation: Removing the Target Network

{VARIANT_NOTES['dqn_no_target']} The final mean return was {mean_ci_text(no_target['final_return_mean'], no_target['final_return_ci95'])}, lower than baseline. It still reached the 200 threshold in {int(no_target['threshold_reach_count'])}/{int(no_target['n'])} seeds, but its median first reach step was {format_steps(no_target['first_reach_step_median'])}, slower than baseline's {format_steps(baseline['first_reach_step_median'])}.

{paired_sentence('dqn_no_target')} The stability diagnostics are the clearest warning sign: the no-target variant averaged {format_number(no_target['collapse_count_mean'], 1)} collapses per run, compared with {format_number(baseline['collapse_count_mean'], 1)} for baseline. This matches the expected role of a target network. Without a slower-moving bootstrap target, the value update target is less stable.

## Ablation: Removing Replay

{VARIANT_NOTES['dqn_no_replay']} This was the clearest failure. Final mean return was {mean_ci_text(no_replay['final_return_mean'], no_replay['final_return_ci95'])}, and zero out of {int(no_replay['n'])} seeds reached the success threshold. Its normalized AUC was {mean_ci_text(no_replay['normalized_auc_mean'], no_replay['normalized_auc_ci95'])}, far below every replay-based variant.

{paired_sentence('dqn_no_replay')} The collapse count is {format_number(no_replay['collapse_count_mean'], 1)}, but that is not a sign of stability. The variant never crossed the threshold, so it had no successful regime from which to collapse. The result shows that replay is not just an implementation detail here; it is the core mechanism that gives DQN enough decorrelated and reused experience to learn LunarLander.

## Extension: Double DQN

{VARIANT_NOTES['double_dqn']} Double DQN produced a final mean return of {mean_ci_text(double['final_return_mean'], double['final_return_ci95'])}, above baseline. It reached the threshold in all seeds and had a final solved count of {int(double['final_solved_count'])}/{int(double['n'])}. Its median first reach step was {format_steps(double['first_reach_step_median'])}, so it was slower to cross the threshold than baseline, but it ended with a better final policy.

{paired_sentence('double_dqn')} The Q-overestimation proxy was {format_number(double['q_overestimation_proxy_last10_mean'], 3)} for Double DQN and {format_number(baseline['q_overestimation_proxy_last10_mean'], 3)} for baseline. This proxy did not decrease, so the performance gain should not be described as proven lower overestimation from this single metric. A safer interpretation is that Double DQN improved final policy quality and reliability even though the logged proxy is sensitive to Q-value scale and late-training state distribution.

## Extension: Dueling DQN

{VARIANT_NOTES['dueling_dqn']} Dueling DQN had the best final mean return: {mean_ci_text(dueling['final_return_mean'], dueling['final_return_ci95'])}. It solved at the end in {int(dueling['final_solved_count'])}/{int(dueling['n'])} seeds and reached the threshold in all seeds. Its average collapse count was {format_number(dueling['collapse_count_mean'], 1)}, lower than the baseline value of {format_number(baseline['collapse_count_mean'], 1)}.

{paired_sentence('dueling_dqn')} The tradeoff is speed. Its median first reach step was {format_steps(dueling['first_reach_step_median'])}, slower than baseline. The result suggests that separating state value from action advantage helped late policy quality and stability more than early learning speed.

## Extension: Double + Dueling DQN

{VARIANT_NOTES['double_dueling_dqn']} The combined variant had final mean return {mean_ci_text(double_dueling['final_return_mean'], double_dueling['final_return_ci95'])}, but with very high seed standard deviation of {format_number(double_dueling['final_return_std'], 1)}. Four seeds finished near or above strong solved performance, while one seed failed badly.

{paired_sentence('double_dueling_dqn')} This means the combination can work very well, but in this result set it is less reliable than using the dueling head alone. The mean hides a bimodal-looking behavior: most seeds are strong, one seed is a clear failure.

![Stability and regression diagnostics](figures/lunarlander/fig_stability_regression.png)

## Optimization, Q-Value, and Behavior Diagnostics

The optimization figure shows that the no-replay variant had the largest late TD loss and a poor final policy. This supports the interpretation that removing replay makes the Bellman regression problem harder and less useful. Dueling DQN had the lowest Q-overestimation proxy among the strong final performers, while Double DQN had high final return even though this proxy was not lower than baseline.

![Optimization and Q diagnostics](figures/lunarlander/fig_optimization_q_diagnostics.png)

Final evaluation behavior summary:

{behavior_table(final_behavior)}

The behavior figure connects return to LunarLander outcomes. Good variants show higher final success and lower crash/timeout rates. Engine-use fractions help distinguish controlled landings from hovering or unstable correction.

![LunarLander behavior diagnostics](figures/lunarlander/fig_lunarlander_behavior.png)

## Main Conclusions

Experience replay is essential in this experiment. Removing replay prevented every seed from reaching the success threshold.

The target network matters for stability. Removing it did not make learning impossible, but it lowered final performance and increased collapse frequency.

Baseline DQN learned early, which gave it the best normalized AUC, but it often failed to retain its best policy until the final checkpoint.

Dueling DQN was the strongest final performer and had fewer collapses than baseline, suggesting better late-training value representation for LunarLander.

Double DQN improved final return, but the logged overestimation proxy alone does not prove a clean reduction in overestimation for these runs.

Double + Dueling DQN showed high upside but worse reliability than Dueling DQN alone because one seed failed badly.
"""
    (analysis_dir / "lunarlander_report_v0.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    analysis_dir = Path(args.analysis_dir)
    figures_dir = analysis_dir / "figures" / "lunarlander"
    tables_dir = analysis_dir / "tables"

    summary_df, eval_df, _update_df = load_data(output_root, args.env_slug)
    variant_summary = build_variant_summary(summary_df)
    paired = build_paired_comparison(summary_df)
    eval_curve = build_eval_curve(eval_df)
    final_behavior = build_final_behavior(eval_df)
    phase_summary = build_phase_summary(eval_df)

    save_tables(tables_dir, summary_df, variant_summary, paired, eval_curve, final_behavior, phase_summary)
    save_figures(figures_dir, summary_df, variant_summary, paired, eval_curve, final_behavior, phase_summary)
    write_metrics_tutorial(analysis_dir)
    write_report(analysis_dir, summary_df, variant_summary, paired, final_behavior)

    print(f"Wrote reports to {analysis_dir / 'metrics_tutorials.md'} and {analysis_dir / 'lunarlander_report_v0.md'}")
    print(f"Wrote figures to {figures_dir}")
    print(f"Wrote tables to {tables_dir}")


if __name__ == "__main__":
    main()
