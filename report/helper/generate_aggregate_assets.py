#!/usr/bin/env python3
"""Generate cross-task aggregate result-analysis assets for the report."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

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
DEFAULT_LUNAR_SUMMARY = REPO_ROOT / "report/helper/tables/lunar_lander_variant_summary.csv"
DEFAULT_FREEWAY_SUMMARY = REPO_ROOT / "report/helper/tables/freeway_variant_summary.csv"
DEFAULT_TABLE_CSV = REPO_ROOT / "report/helper/tables/aggregate_variant_comparison.csv"
DEFAULT_TABLE_TEX = REPO_ROOT / "report/helper/tables/aggregate_variant_comparison.tex"
DEFAULT_FIGURE = REPO_ROOT / "report/helper/figures/aggregate_cross_task_stability.png"

LUNAR_THRESHOLD = 200.0
FREEWAY_THRESHOLD = 15.0

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
    "lunar": "#0033A0",
    "freeway": "#009E73",
    "mean": "#8A8D8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lunar-summary", type=Path, default=DEFAULT_LUNAR_SUMMARY)
    parser.add_argument("--freeway-summary", type=Path, default=DEFAULT_FREEWAY_SUMMARY)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--summary-tex", type=Path, default=DEFAULT_TABLE_TEX)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    return parser.parse_args()


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


def variant_sort_key(value: str) -> int:
    try:
        return VARIANT_ORDER.index(value)
    except ValueError:
        return len(VARIANT_ORDER)


def build_aggregate_summary(lunar: pd.DataFrame, freeway: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant in VARIANT_ORDER:
        lunar_row = lunar.loc[lunar["variant"] == variant]
        freeway_row = freeway.loc[freeway["variant"] == variant]
        if lunar_row.empty or freeway_row.empty:
            continue

        lunar_values = lunar_row.iloc[0]
        freeway_values = freeway_row.iloc[0]
        lunar_final_norm = float(lunar_values["final_eval_return_mean"]) / LUNAR_THRESHOLD
        freeway_final_norm = float(freeway_values["final_eval_return_mean"]) / FREEWAY_THRESHOLD
        lunar_sustain_frac = float(lunar_values["sustained_200_count"]) / float(lunar_values["seeds"])
        freeway_sustain_frac = float(freeway_values["sustained_15_count"]) / float(freeway_values["seeds"])
        lunar_gap_norm = float(lunar_values["peak_to_final_gap_mean"]) / LUNAR_THRESHOLD
        freeway_gap_norm = float(freeway_values["peak_to_final_gap_mean"]) / FREEWAY_THRESHOLD

        rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "lunar_final_threshold_ratio": lunar_final_norm,
                "freeway_final_threshold_ratio": freeway_final_norm,
                "mean_final_threshold_ratio": float(np.mean([lunar_final_norm, freeway_final_norm])),
                "lunar_sustained_fraction": lunar_sustain_frac,
                "freeway_sustained_fraction": freeway_sustain_frac,
                "mean_sustained_fraction": float(np.mean([lunar_sustain_frac, freeway_sustain_frac])),
                "lunar_peak_gap_threshold_ratio": lunar_gap_norm,
                "freeway_peak_gap_threshold_ratio": freeway_gap_norm,
                "mean_peak_gap_threshold_ratio": float(np.mean([lunar_gap_norm, freeway_gap_norm])),
            }
        )

    summary = pd.DataFrame(rows)
    summary["variant_order"] = summary["variant"].map(variant_sort_key)
    return summary.sort_values("variant_order").drop(columns=["variant_order"])


def format_ratio(value: float) -> str:
    return f"{value:.2f}"


def format_percent(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def write_summary_tex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{adjustbox}{width=\textwidth}",
        r"\begin{tabular}{@{}lccccccc@{}}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Lunar Final/Thr.} & \textbf{Freeway Final/Thr.} & "
        r"\textbf{Mean Final/Thr.} & \textbf{Lunar Sustained} & \textbf{Freeway Sustained} & "
        r"\textbf{Mean Sustained} & \textbf{Mean Peak Gap/Thr.} \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    latex_escape(row.variant_label),
                    format_ratio(row.lunar_final_threshold_ratio),
                    format_ratio(row.freeway_final_threshold_ratio),
                    format_ratio(row.mean_final_threshold_ratio),
                    format_percent(row.lunar_sustained_fraction),
                    format_percent(row.freeway_sustained_fraction),
                    format_percent(row.mean_sustained_fraction),
                    format_ratio(row.mean_peak_gap_threshold_ratio),
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


def draw_cross_task_stability(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(summary))
    labels = summary["variant_label"].to_list()
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3), dpi=300)

    ax = axes[0]
    ax.bar(
        x - width / 2,
        summary["lunar_final_threshold_ratio"],
        width,
        label="Lunar Lander",
        color=COLORS["lunar"],
    )
    ax.bar(
        x + width / 2,
        summary["freeway_final_threshold_ratio"],
        width,
        label="Freeway",
        color=COLORS["freeway"],
    )
    ax.axhline(1.0, color="#111111", linestyle="--", linewidth=1.0)
    ax.set_title("Final performance relative to threshold", fontsize=11, weight="bold")
    ax.set_ylabel("Final evaluation / task threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(-1.5, 2.2)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    ax.bar(
        x - width / 2,
        summary["lunar_peak_gap_threshold_ratio"],
        width,
        label="Lunar Lander",
        color=COLORS["lunar"],
    )
    ax.bar(
        x + width / 2,
        summary["freeway_peak_gap_threshold_ratio"],
        width,
        label="Freeway",
        color=COLORS["freeway"],
    )
    ax.set_title("Policy loss after best checkpoint", fontsize=11, weight="bold")
    ax.set_ylabel("Peak-to-final gap / task threshold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 2.1)
    set_axis_style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    lunar = pd.read_csv(args.lunar_summary)
    freeway = pd.read_csv(args.freeway_summary)
    summary = build_aggregate_summary(lunar, freeway)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_csv, index=False)
    write_summary_tex(summary, args.summary_tex)
    draw_cross_task_stability(summary, args.figure)


if __name__ == "__main__":
    main()
