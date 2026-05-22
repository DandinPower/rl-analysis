#!/usr/bin/env python3
"""Generate Lunar Lander result-analysis assets for the report."""

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
DEFAULT_RESULT_ROOT = REPO_ROOT / "output/full_lunarlander_10seeds_20260516_042845/lunarlander_v3"
DEFAULT_TABLE_CSV = REPO_ROOT / "report/helper/tables/lunar_lander_variant_summary.csv"
DEFAULT_TABLE_TEX = REPO_ROOT / "report/helper/tables/lunar_lander_variant_summary.tex"
DEFAULT_LEARNING_FIGURE = REPO_ROOT / "report/helper/figures/lunar_lander_learning_curves.png"
DEFAULT_BEHAVIOR_FIGURE = REPO_ROOT / "report/helper/figures/lunar_lander_behavior_breakdown.png"
DEFAULT_RETENTION_FIGURE = REPO_ROOT / "report/helper/figures/lunar_lander_retention_gap.png"
DEFAULT_STRESS_FIGURE = REPO_ROOT / "report/helper/figures/lunar_lander_optimization_stress.png"

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
        variant = path.parts[-4]
        seed = int(path.parts[-3].split("_")[-1])
        for record in read_jsonl(path):
            lunar = record.get("lunarlander_eval", {})
            rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "seed": seed,
                    "step": int(record["global_env_step"]),
                    "eval_index": int(record["eval_index"]),
                    "return_mean": float(record["return_mean"]),
                    "return_std": float(record.get("return_std", np.nan)),
                    "episode_length_mean": float(record.get("episode_length_mean", np.nan)),
                    "success_rate": float(lunar.get("success_rate", np.nan)),
                    "landing_success_rate": float(lunar.get("landing_success_rate", np.nan)),
                    "crash_rate": float(lunar.get("crash_rate", np.nan)),
                    "timeout_rate": float(lunar.get("timeout_rate", np.nan)),
                    "main_engine_fraction": float(lunar.get("main_engine_action_fraction_mean", np.nan)),
                    "side_engine_fraction": float(lunar.get("side_engine_action_fraction_mean", np.nan)),
                    "action_entropy": float(record.get("action_entropy_mean", np.nan)),
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
        variant = payload["variant"]
        performance = payload.get("performance", {})
        efficiency = payload.get("sample_efficiency", {})
        stability = payload.get("stability", {})
        diagnostics = payload.get("failure_diagnostics", {})
        q_diagnostics = payload.get("q_diagnostics", {})
        optimization = payload.get("optimization", {})
        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": int(payload["seed"]),
                "best_eval_return": float(performance.get("best_eval_return_mean", np.nan)),
                "final_eval_return": float(performance.get("final_eval_return_mean", np.nan)),
                "first_step_reaching_threshold": efficiency.get("first_step_reaching_threshold"),
                "first_step_sustained_threshold": efficiency.get("first_step_sustained_threshold"),
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
    frame["first_step_reaching_threshold"] = pd.to_numeric(
        frame["first_step_reaching_threshold"], errors="coerce"
    )
    frame["first_step_sustained_threshold"] = pd.to_numeric(
        frame["first_step_sustained_threshold"], errors="coerce"
    )
    frame["peak_to_final_gap"] = frame["best_eval_return"] - frame["final_eval_return"]
    return frame


def build_variant_summary(eval_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    final_eval = eval_df.loc[eval_df.groupby(["variant", "seed"])["step"].idxmax()].copy()
    final_behavior = (
        final_eval.groupby("variant", as_index=False)
        .agg(
            final_success_rate=("success_rate", "mean"),
            final_landing_success_rate=("landing_success_rate", "mean"),
            final_crash_rate=("crash_rate", "mean"),
            final_timeout_rate=("timeout_rate", "mean"),
            final_main_engine_fraction=("main_engine_fraction", "mean"),
            final_side_engine_fraction=("side_engine_fraction", "mean"),
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
            first_200_mean=("first_step_reaching_threshold", "mean"),
            first_200_count=("first_step_reaching_threshold", "count"),
            sustained_200_mean=("first_step_sustained_threshold", "mean"),
            sustained_200_count=("first_step_sustained_threshold", "count"),
            peak_to_final_gap_mean=("peak_to_final_gap", "mean"),
            largest_eval_drop_mean=("largest_eval_drop", "mean"),
            q_spike_rate=("q_value_explosion", "mean"),
            max_q_value_seen_mean=("max_q_value_seen", "mean"),
            td_error_abs_mean_last_10pct=("td_error_abs_mean_last_10pct", "mean"),
            catastrophic_collapse_count_mean=("catastrophic_collapse_count", "mean"),
        )
        .merge(final_behavior, on="variant", how="left")
    )
    summary["variant_label"] = summary["variant"].map(VARIANT_LABELS).fillna(summary["variant"])
    summary["variant_order"] = summary["variant"].map(lambda value: variant_sort_key(str(value)))
    return summary.sort_values("variant_order").drop(columns=["variant_order"])


def format_mean_std(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.1f} $\\pm$ {std_value:.1f}"


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
        r"\textbf{Reached 200} & \textbf{Sustained 200} & \textbf{First 200} & "
        r"\textbf{Final Success} & \textbf{Peak Gap} & \textbf{Largest Drop} & \textbf{Q Spike} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        reached = f"{int(row.first_200_count)}/{int(row.seeds)}"
        sustained = f"{int(row.sustained_200_count)}/{int(row.seeds)}"
        final_success = f"{100 * row.final_success_rate:.1f}\\%"
        q_spike = f"{100 * row.q_spike_rate:.0f}\\%"
        lines.append(
            " & ".join(
                [
                    latex_escape(row.variant_label),
                    format_mean_std(row.best_eval_return_mean, row.best_eval_return_std),
                    format_mean_std(row.final_eval_return_mean, row.final_eval_return_std),
                    reached,
                    sustained,
                    format_step(row.first_200_mean),
                    final_success,
                    f"{row.peak_to_final_gap_mean:.1f}",
                    f"{row.largest_eval_drop_mean:.1f}",
                    q_spike,
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
            .agg(mean_return=("return_mean", "mean"), std_return=("return_mean", "std"), seeds=("seed", "nunique"))
            .sort_values("step")
        )
        stderr = grouped["std_return"].to_numpy() / np.sqrt(grouped["seeds"].to_numpy())
        steps = grouped["step"].to_numpy() / 1000
        mean_return = grouped["mean_return"].to_numpy()
        color = COLORS.get(variant, "#333333")
        ax.plot(steps, mean_return, label=VARIANT_LABELS.get(variant, variant), color=color, linewidth=2.0)
        ax.fill_between(steps, mean_return - stderr, mean_return + stderr, color=color, alpha=0.12, linewidth=0)

    ax.axhline(200, color="#111111", linestyle="--", linewidth=1.1, label="Success threshold")
    ax.set_title("Lunar Lander evaluation return over training", fontsize=12, weight="bold")
    ax.set_xlabel("Environment steps (thousands)")
    ax.set_ylabel("Mean evaluation return")
    ax.set_xlim(10, 500)
    ax.set_ylim(-320, 280)
    set_axis_style(ax)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_behavior_breakdown(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.copy()
    x = np.arange(len(ordered))
    labels = ordered["variant_label"].to_list()

    landing = ordered["final_landing_success_rate"].to_numpy()
    crash = ordered["final_crash_rate"].to_numpy()
    timeout = ordered["final_timeout_rate"].to_numpy()
    other = np.clip(1.0 - landing - crash - timeout, 0.0, 1.0)
    main = ordered["final_main_engine_fraction"].to_numpy()
    side = ordered["final_side_engine_fraction"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), dpi=300)
    ax = axes[0]
    ax.bar(x, landing * 100, label="Successful landing", color="#009E73")
    ax.bar(x, timeout * 100, bottom=landing * 100, label="Timeout", color="#56B4E9")
    ax.bar(x, crash * 100, bottom=(landing + timeout) * 100, label="Crash", color="#D55E00")
    ax.bar(x, other * 100, bottom=(landing + timeout + crash) * 100, label="Other", color="#8A8D8F")
    ax.set_title("Final episode outcomes", fontsize=11, weight="bold")
    ax.set_ylabel("Share of evaluation episodes (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, main * 100, width, label="Main engine", color="#0033A0")
    ax.bar(x + width / 2, side * 100, width, label="Side engines", color="#CC79A7")
    ax.set_title("Final engine-use pattern", fontsize=11, weight="bold")
    ax.set_ylabel("Action fraction (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 65)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

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
    ax.set_title("Return lost after the best checkpoint", fontsize=11, weight="bold")
    ax.set_ylabel("Best return minus final return")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    set_axis_style(ax)

    ax = axes[1]
    width = 0.36
    ax.bar(x - width / 2, ordered["first_200_count"], width, label="Reached 200", color="#0072B2")
    ax.bar(x + width / 2, ordered["sustained_200_count"], width, label="Sustained 200", color="#009E73")
    ax.set_title("Threshold reach versus retention", fontsize=11, weight="bold")
    ax.set_ylabel("Seeds out of 10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 10.5)
    ax.set_yticks(np.arange(0, 11, 2))
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
    ax.set_yscale("log")
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
    ax.bar(x - width / 2, ordered["q_spike_rate"] * 100, width, label="Large Q spike", color="#D55E00")
    ax.bar(
        x + width / 2,
        ordered["catastrophic_collapse_count_mean"],
        width,
        label="Collapse count",
        color="#8A8D8F",
    )
    ax.set_title("Instability indicators", fontsize=11, weight="bold")
    ax.set_ylabel("Rate (%) or count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

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
    draw_behavior_breakdown(summary, args.behavior_figure)
    draw_retention_gap(summary, args.retention_figure)
    draw_optimization_stress(summary, args.stress_figure)


if __name__ == "__main__":
    main()
