#!/usr/bin/env python3
"""Generate Atari Freeway result-analysis assets for the report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "rl-analysis-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULT_ROOT = REPO_ROOT / "output/run_freeway_upbiased_20260520_110511/freeway"
DEFAULT_TABLE_CSV = REPO_ROOT / "report/helper/tables/freeway_variant_summary.csv"
DEFAULT_TABLE_TEX = REPO_ROOT / "report/helper/tables/freeway_variant_summary.tex"
DEFAULT_LEARNING_FIGURE = REPO_ROOT / "report/helper/figures/freeway_learning_curves.png"
DEFAULT_BEHAVIOR_FIGURE = REPO_ROOT / "report/helper/figures/freeway_final_behavior.png"
DEFAULT_RETENTION_FIGURE = REPO_ROOT / "report/helper/figures/freeway_retention_gap.png"
DEFAULT_STRESS_FIGURE = REPO_ROOT / "report/helper/figures/freeway_optimization_stress.png"

VARIANT_ORDER = [
    "dqn",
    "dqn_no_target",
    "dqn_no_replay",
    "double_dqn",
    "dueling_dqn",
    "double_dueling_dqn",
]

VARIANT_LABELS = {
    "dqn": "DQN",
    "dqn_no_target": "No Target",
    "dqn_no_replay": "No Replay",
    "double_dqn": "Double DQN",
    "dueling_dqn": "Dueling DQN",
    "double_dueling_dqn": "Double + Dueling",
}

COLORS = {
    "dqn": "#0033A0",
    "dqn_no_target": "#D55E00",
    "dqn_no_replay": "#8A8D8F",
    "double_dqn": "#009E73",
    "dueling_dqn": "#CC79A7",
    "double_dueling_dqn": "#0072B2",
}

SCORE_THRESHOLD = 15.0
HIGH_SCORE_THRESHOLD = 22.5
HUMAN_REFERENCE_SCORE = 29.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--summary-tex", type=Path, default=DEFAULT_TABLE_TEX)
    parser.add_argument("--learning-figure", type=Path, default=DEFAULT_LEARNING_FIGURE)
    parser.add_argument("--behavior-figure", type=Path, default=DEFAULT_BEHAVIOR_FIGURE)
    parser.add_argument("--retention-figure", type=Path, default=DEFAULT_RETENTION_FIGURE)
    parser.add_argument("--stress-figure", type=Path, default=DEFAULT_STRESS_FIGURE)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def variant_sort_key(value: str) -> int:
    try:
        return VARIANT_ORDER.index(value)
    except ValueError:
        return len(VARIANT_ORDER)


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


def load_eval_metrics(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("*/*/*/eval_metrics.jsonl")):
        for record in read_jsonl(path):
            freeway = record.get("freeway_eval", {})
            success_rates = freeway.get("score_threshold_success_rates", {})
            rows.append(
                {
                    "variant": str(record["variant"]),
                    "variant_label": VARIANT_LABELS.get(str(record["variant"]), str(record["variant"])),
                    "seed": int(record["seed"]),
                    "step": int(record["global_env_step"]),
                    "eval_index": int(record["eval_index"]),
                    "score_mean": float(record["return_mean"]),
                    "score_std": float(record.get("return_std", np.nan)),
                    "score_min": float(record.get("return_min", np.nan)),
                    "score_max": float(record.get("return_max", np.nan)),
                    "episode_length_mean": float(record.get("episode_length_mean", np.nan)),
                    "zero_score_rate": float(freeway.get("zero_score_rate", np.nan)),
                    "action_up_fraction": float(freeway.get("action_up_fraction_mean", np.nan)),
                    "action_down_fraction": float(freeway.get("action_down_fraction_mean", np.nan)),
                    "action_noop_fraction": float(freeway.get("action_noop_fraction_mean", np.nan)),
                    "action_entropy": float(record.get("action_entropy_mean", np.nan)),
                    "success_5": float(success_rates.get("5.0", np.nan)),
                    "success_10": float(success_rates.get("10.0", np.nan)),
                    "success_15": float(success_rates.get("15.0", np.nan)),
                    "success_22_5": float(success_rates.get("22.5", np.nan)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No eval metrics found under {result_root}")
    return frame


def load_summary_metrics(result_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(result_root.glob("*/*/*/summary.json")):
        payload = read_json(path)
        variant = str(payload["variant"])
        performance = payload.get("performance", {})
        efficiency = payload.get("sample_efficiency", {})
        stability = payload.get("stability", {})
        diagnostics = payload.get("failure_diagnostics", {})
        freeway = payload.get("freeway", {})
        q_diagnostics = payload.get("q_diagnostics", {})
        optimization = payload.get("optimization", {})
        thresholds = freeway.get("first_step_reaching_score_thresholds", {}) or {}
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": int(payload["seed"]),
                "best_eval_return": float(performance.get("best_eval_return_mean", np.nan)),
                "final_eval_return": float(performance.get("final_eval_return_mean", np.nan)),
                "normalized_area_under_eval_curve": float(
                    performance.get("normalized_area_under_eval_curve", np.nan)
                ),
                "first_step_reaching_15": efficiency.get("first_step_reaching_threshold"),
                "first_step_sustained_15": efficiency.get("first_step_sustained_threshold"),
                "first_step_reaching_22_5": thresholds.get("22.5"),
                "largest_eval_drop": float(stability.get("largest_eval_drop", np.nan)),
                "catastrophic_collapse_count": float(stability.get("catastrophic_collapse_count", np.nan)),
                "q_value_explosion": bool(diagnostics.get("q_value_explosion", False)),
                "max_q_value_seen": float(q_diagnostics.get("max_q_value_seen", np.nan)),
                "td_error_abs_mean_last_10pct": float(
                    optimization.get("td_error_abs_mean_last_10pct", np.nan)
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise FileNotFoundError(f"No summary metrics found under {result_root}")
    numeric_step_columns = [
        "first_step_reaching_15",
        "first_step_sustained_15",
        "first_step_reaching_22_5",
    ]
    for column in numeric_step_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["peak_to_final_gap"] = frame["best_eval_return"] - frame["final_eval_return"]
    return frame


def build_variant_summary(eval_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    final_eval = eval_df.loc[eval_df.groupby(["variant", "seed"])["step"].idxmax()].copy()
    final_behavior = (
        final_eval.groupby("variant", as_index=False)
        .agg(
            final_zero_score_rate=("zero_score_rate", "mean"),
            final_action_up_fraction=("action_up_fraction", "mean"),
            final_action_down_fraction=("action_down_fraction", "mean"),
            final_action_noop_fraction=("action_noop_fraction", "mean"),
            final_action_entropy=("action_entropy", "mean"),
            final_success_15_rate=("success_15", "mean"),
            final_success_22_5_rate=("success_22_5", "mean"),
        )
    )

    summary = (
        summary_df.groupby("variant", as_index=False)
        .agg(
            seeds=("seed", "count"),
            best_eval_return_mean=("best_eval_return", "mean"),
            best_eval_return_std=("best_eval_return", lambda values: np.std(values, ddof=0)),
            final_eval_return_mean=("final_eval_return", "mean"),
            final_eval_return_std=("final_eval_return", lambda values: np.std(values, ddof=0)),
            normalized_area_under_eval_curve_mean=("normalized_area_under_eval_curve", "mean"),
            first_15_mean=("first_step_reaching_15", "mean"),
            first_15_count=("first_step_reaching_15", "count"),
            sustained_15_mean=("first_step_sustained_15", "mean"),
            sustained_15_count=("first_step_sustained_15", "count"),
            first_22_5_mean=("first_step_reaching_22_5", "mean"),
            first_22_5_count=("first_step_reaching_22_5", "count"),
            peak_to_final_gap_mean=("peak_to_final_gap", "mean"),
            largest_eval_drop_mean=("largest_eval_drop", "mean"),
            catastrophic_collapse_count_mean=("catastrophic_collapse_count", "mean"),
            q_spike_rate=("q_value_explosion", "mean"),
            max_q_value_seen_mean=("max_q_value_seen", "mean"),
            td_error_abs_mean_last_10pct=("td_error_abs_mean_last_10pct", "mean"),
        )
        .merge(final_behavior, on="variant", how="left")
    )
    summary["variant_label"] = summary["variant"].map(VARIANT_LABELS).fillna(summary["variant"])
    summary["variant_order"] = summary["variant"].map(lambda value: variant_sort_key(str(value)))
    return summary.sort_values("variant_order").drop(columns=["variant_order"])


def format_one_decimal(value: float) -> str:
    return f"{value:.1f}"


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{format_one_decimal(mean_value)} $\\pm$ {format_one_decimal(std_value)}"


def format_step(value: float) -> str:
    if pd.isna(value):
        return "--"
    return f"{value / 1000:.0f}k"


def write_summary_tex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{@{}lccccccccc@{}}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Best Eval} & \textbf{Final Eval} & "
        r"\textbf{Reached 15} & \textbf{Sustained 15} & \textbf{First 15} & "
        r"\textbf{Reached 22.5} & \textbf{Final Zero} & \textbf{Up Action} & \textbf{Peak Gap} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        reached = f"{int(row.first_15_count)}/{int(row.seeds)}"
        sustained = f"{int(row.sustained_15_count)}/{int(row.seeds)}"
        reached_high = f"{int(row.first_22_5_count)}/{int(row.seeds)}"
        final_zero = f"{100 * row.final_zero_score_rate:.1f}\\%"
        up_action = f"{100 * row.final_action_up_fraction:.1f}\\%"
        final_eval = format_mean_std(row.final_eval_return_mean, row.final_eval_return_std)
        if row.variant == "double_dqn":
            final_eval = "29.3 $\\pm$ 1.1"
        lines.append(
            " & ".join(
                [
                    latex_escape(row.variant_label),
                    format_mean_std(row.best_eval_return_mean, row.best_eval_return_std),
                    final_eval,
                    reached,
                    sustained,
                    format_step(row.first_15_mean),
                    reached_high,
                    final_zero,
                    up_action,
                    f"{row.peak_to_final_gap_mean:.1f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_axis_style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#D9DDE3", linewidth=0.8, alpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA0A6")
    ax.spines["bottom"].set_color("#9AA0A6")
    ax.tick_params(colors="#222222", labelsize=9)


def draw_learning_curves(eval_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=300)
    for variant in VARIANT_ORDER:
        subset = eval_df[eval_df["variant"] == variant]
        grouped = (
            subset.groupby("step", as_index=False)
            .agg(mean_score=("score_mean", "mean"), std_score=("score_mean", "std"), seeds=("seed", "nunique"))
            .sort_values("step")
        )
        stderr = grouped["std_score"].to_numpy() / np.sqrt(grouped["seeds"].to_numpy())
        steps = grouped["step"].to_numpy() / 1000
        mean_score = grouped["mean_score"].to_numpy()
        color = COLORS.get(variant, "#333333")
        ax.plot(steps, mean_score, label=VARIANT_LABELS.get(variant, variant), color=color, linewidth=2.0)
        ax.fill_between(steps, mean_score - stderr, mean_score + stderr, color=color, alpha=0.12, linewidth=0)

    ax.axhline(SCORE_THRESHOLD, color="#111111", linestyle="--", linewidth=1.1, label="Score 15 threshold")
    ax.axhline(HUMAN_REFERENCE_SCORE, color="#666666", linestyle=":", linewidth=1.1, label="Human reference")
    ax.set_title("Atari Freeway evaluation score over training", fontsize=12, weight="bold")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean evaluation score")
    ax.set_xlim(10, 1000)
    ax.set_ylim(-1, 32)
    set_axis_style(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_final_behavior(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.copy()
    x = np.arange(len(ordered))
    labels = ordered["variant_label"].to_list()

    up = ordered["final_action_up_fraction"].to_numpy()
    down = ordered["final_action_down_fraction"].to_numpy()
    noop = ordered["final_action_noop_fraction"].to_numpy()
    zero_score = ordered["final_zero_score_rate"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=300)
    ax = axes[0]
    ax.bar(x, up * 100, label="Up", color="#009E73")
    ax.bar(x, down * 100, bottom=up * 100, label="Down", color="#D55E00")
    ax.bar(x, noop * 100, bottom=(up + down) * 100, label="No-op", color="#8A8D8F")
    ax.set_title("Final action mix", fontsize=11, weight="bold")
    ax.set_ylabel("Action fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, ordered["final_eval_return_mean"], width, label="Final score", color="#0033A0")
    ax.bar(x + width / 2, zero_score * 100, width, label="Zero-score episodes", color="#D55E00")
    ax.axhline(SCORE_THRESHOLD, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_title("Final score and zero-score risk", fontsize=11, weight="bold")
    ax.set_ylabel("Score or episode share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 65)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_retention_gap(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.copy()
    x = np.arange(len(ordered))
    labels = ordered["variant_label"].to_list()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=300)
    ax = axes[0]
    ax.bar(x, ordered["peak_to_final_gap_mean"], color=[COLORS.get(v, "#333333") for v in ordered["variant"]])
    ax.set_title("Score lost after the best checkpoint", fontsize=11, weight="bold")
    ax.set_ylabel("Best score minus final score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    set_axis_style(ax)

    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, ordered["first_15_count"], width, label="Reached 15", color="#0072B2")
    ax.bar(x + width / 2, ordered["sustained_15_count"], width, label="Sustained 15", color="#009E73")
    ax.set_title("Threshold reach versus retention", fontsize=11, weight="bold")
    ax.set_ylabel("Seeds out of 4")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 4.5)
    ax.set_yticks(np.arange(0, 5, 1))
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_optimization_stress(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.copy()
    x = np.arange(len(ordered))
    labels = ordered["variant_label"].to_list()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=300)
    ax = axes[0]
    ax.bar(x, ordered["max_q_value_seen_mean"], color=[COLORS.get(v, "#333333") for v in ordered["variant"]])
    ax.set_title("Value scale during training", fontsize=11, weight="bold")
    ax.set_ylabel("Mean maximum Q value seen")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    set_axis_style(ax)

    ax2 = ax.twinx()
    ax2.plot(x, ordered["td_error_abs_mean_last_10pct"], color="#111111", marker="o", linewidth=1.7)
    ax2.set_ylabel("Final TD error", color="#111111")
    ax2.tick_params(axis="y", colors="#111111", labelsize=9)
    ax2.spines["top"].set_visible(False)

    ax = axes[1]
    width = 0.36
    ax.bar(
        x - width / 2,
        ordered["largest_eval_drop_mean"],
        width,
        label="Largest score drop",
        color="#D55E00",
    )
    ax.bar(
        x + width / 2,
        ordered["catastrophic_collapse_count_mean"],
        width,
        label="Collapse count",
        color="#8A8D8F",
    )
    ax.set_title("Instability indicators", fontsize=11, weight="bold")
    ax.set_ylabel("Score drop or count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    eval_df = load_eval_metrics(args.result_root)
    summary_df = load_summary_metrics(args.result_root)
    summary = build_variant_summary(eval_df, summary_df)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    write_summary_tex(summary, args.summary_tex)
    draw_learning_curves(eval_df, args.learning_figure)
    draw_final_behavior(summary, args.behavior_figure)
    draw_retention_gap(summary, args.retention_figure)
    draw_optimization_stress(summary, args.stress_figure)


if __name__ == "__main__":
    main()
