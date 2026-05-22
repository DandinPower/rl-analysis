#!/usr/bin/env python3
"""Generate algorithm-selection and ablation-design assets for the report."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import os

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
DEFAULT_CSV = REPO_ROOT / "report/helper/tables/algorithm_ablation_design.csv"
DEFAULT_TEX = REPO_ROOT / "report/helper/tables/algorithm_ablation_design.tex"
DEFAULT_FIGURE = REPO_ROOT / "report/helper/figures/algorithm_ablation_matrix.png"


@dataclass(frozen=True)
class VariantDesign:
    variant: str
    target_network: bool
    replay_buffer: bool
    double_update: bool
    dueling_head: bool
    role: str
    expected_effect: str


VARIANTS = (
    VariantDesign(
        variant="DQN",
        target_network=True,
        replay_buffer=True,
        double_update=False,
        dueling_head=False,
        role="Baseline with replay and a separate target network.",
        expected_effect="Set the reference for learning speed and stability in both tasks.",
    ),
    VariantDesign(
        variant="DQN without target network",
        target_network=False,
        replay_buffer=True,
        double_update=False,
        dueling_head=False,
        role="Ablation of the slow-moving bootstrap target.",
        expected_effect="More oscillation and weaker policy retention because the target changes every update.",
    ),
    VariantDesign(
        variant="DQN without replay buffer",
        target_network=True,
        replay_buffer=False,
        double_update=False,
        dueling_head=False,
        role="Ablation of decorrelated sampling and transition reuse.",
        expected_effect="Less stable learning from recent correlated data, especially when rewards are sparse.",
    ),
    VariantDesign(
        variant="Double DQN",
        target_network=True,
        replay_buffer=True,
        double_update=True,
        dueling_head=False,
        role="Extension that separates next-action selection from next-action evaluation.",
        expected_effect="Lower overestimation risk and better final reliability than the baseline.",
    ),
    VariantDesign(
        variant="Dueling DQN",
        target_network=True,
        replay_buffer=True,
        double_update=False,
        dueling_head=True,
        role="Extension that predicts state value and action advantages with separate streams.",
        expected_effect="Better value learning when several actions have similar value in the same state.",
    ),
    VariantDesign(
        variant="Double + Dueling DQN",
        target_network=True,
        replay_buffer=True,
        double_update=True,
        dueling_head=True,
        role="Combined extension that tests whether the two improvements are complementary.",
        expected_effect="Potentially stronger stability and value estimates, but not guaranteed additive gains.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX)
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


def build_design_frame() -> pd.DataFrame:
    rows = []
    for item in VARIANTS:
        rows.append(
            {
                "variant": item.variant,
                "target_network": item.target_network,
                "replay_buffer": item.replay_buffer,
                "double_dqn_update": item.double_update,
                "dueling_network_head": item.dueling_head,
                "role": item.role,
                "expected_effect": item.expected_effect,
            }
        )
    return pd.DataFrame(rows)


def write_table_tex(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    display = df.copy()
    for column in ["target_network", "replay_buffer", "double_dqn_update", "dueling_network_head"]:
        display[column] = np.where(display[column], "Yes", "No")

    headers = [
        "Variant",
        "Target",
        "Replay",
        "Double",
        "Dueling",
        "Design Role and Expected Effect",
    ]
    lines = [
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{0.19\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}"
        r">{\centering\arraybackslash}p{0.07\textwidth}Y@{}}",
        r"\toprule",
        " & ".join(r"\textbf{" + header + "}" for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in display.itertuples(index=False):
        design_note = f"{row.role} {row.expected_effect}"
        lines.append(
            " & ".join(
                [
                    latex_escape(row.variant),
                    latex_escape(row.target_network),
                    latex_escape(row.replay_buffer),
                    latex_escape(row.double_dqn_update),
                    latex_escape(row.dueling_network_head),
                    latex_escape(design_note),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_matrix(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    component_columns = ["target_network", "replay_buffer", "double_dqn_update", "dueling_network_head"]
    component_labels = ["Target network", "Replay buffer", "Double update", "Dueling head"]
    matrix = df[component_columns].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.0, 3.8), dpi=300)
    cmap = matplotlib.colors.ListedColormap(["#F4F5F7", "#0033A0"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(component_labels)))
    ax.set_xticklabels(component_labels, fontsize=9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["variant"], fontsize=9)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Algorithm component matrix", fontsize=12, pad=12, weight="bold")

    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = bool(matrix[row_index, col_index])
            label = "Yes" if value else "No"
            color = "white" if value else "#4A4A4A"
            ax.text(col_index, row_index, label, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(np.arange(-0.5, len(component_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(df), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    df = build_design_frame()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    write_table_tex(df, args.tex)
    draw_matrix(df, args.figure)
    print(f"Wrote {args.csv.relative_to(REPO_ROOT)}")
    print(f"Wrote {args.tex.relative_to(REPO_ROOT)}")
    print(f"Wrote {args.figure.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
