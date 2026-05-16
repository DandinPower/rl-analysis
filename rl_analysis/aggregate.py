"""Aggregate completed experiment runs."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rl_analysis.config import VARIANT_NAMES, get_env_profile
from rl_analysis.utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DQN experiment summaries.")
    parser.add_argument("--env", default="LunarLander-v3")
    parser.add_argument("--output-root", default="output")
    return parser.parse_args()


def _latest_seed_runs(variant_dir: Path) -> list[Path]:
    runs = []
    for seed_dir in sorted(variant_dir.glob("seed_*")):
        candidates = [
            run_dir
            for run_dir in sorted(seed_dir.iterdir())
            if run_dir.is_dir() and (run_dir / "summary.json").exists()
        ]
        if not candidates:
            continue
        runs.append(max(candidates, key=lambda path: path.stat().st_mtime))
    return runs


def _load_summary(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
    }


def aggregate_variant(env_slug: str, variant: str, output_root: Path) -> dict[str, Any] | None:
    variant_dir = output_root / env_slug / variant
    run_dirs = _latest_seed_runs(variant_dir)
    summaries = [_load_summary(run_dir / "summary.json") for run_dir in run_dirs]
    summaries = [summary for summary in summaries if summary.get("training_completed")]
    if not summaries:
        return None

    seeds = [int(summary["seed"]) for summary in summaries]
    final_eval = [
        summary["performance"]["final_eval_return_mean"]
        for summary in summaries
        if summary["performance"]["final_eval_return_mean"] is not None
    ]
    best_eval = [
        summary["performance"]["best_eval_return_mean"]
        for summary in summaries
        if summary["performance"]["best_eval_return_mean"] is not None
    ]
    auc = [
        summary["performance"]["area_under_eval_curve"]
        for summary in summaries
        if summary["performance"]["area_under_eval_curve"] is not None
    ]
    reached_steps = [
        summary["sample_efficiency"]["first_step_reaching_threshold"]
        for summary in summaries
        if summary["sample_efficiency"]["first_step_reaching_threshold"] is not None
    ]
    collapse_counts = [summary["stability"]["catastrophic_collapse_count"] for summary in summaries]
    largest_drops = [
        summary["stability"]["largest_eval_drop"]
        for summary in summaries
        if summary["stability"]["largest_eval_drop"] is not None
    ]

    aggregate = {
        "env_id": summaries[0]["env_id"],
        "variant": variant,
        "num_seeds": len(seeds),
        "seeds": sorted(seeds),
        "final_eval_return": _stats(final_eval),
        "best_eval_return": _stats(best_eval),
        "area_under_eval_curve": _stats(auc),
        "steps_to_threshold": {
            "threshold": summaries[0]["sample_efficiency"]["threshold"],
            **_stats([float(x) for x in reached_steps]),
            "success_rate": float(len(reached_steps) / len(summaries)),
            "num_seeds_reached": len(reached_steps),
        },
        "stability": {
            "catastrophic_collapse_count_mean": float(np.mean(collapse_counts)) if collapse_counts else None,
            "largest_eval_drop_mean": float(np.mean(largest_drops)) if largest_drops else None,
            "eval_return_across_seed_std_mean": float(np.std(final_eval)) if final_eval else None,
        },
        "failure_rate": {
            "nan_rate": float(np.mean([summary["failure_diagnostics"]["nan_detected"] for summary in summaries])),
            "q_value_explosion_rate": float(
                np.mean([summary["failure_diagnostics"]["q_value_explosion"] for summary in summaries])
            ),
            "early_termination_rate": float(
                np.mean([summary["failure_diagnostics"]["early_terminated"] for summary in summaries])
            ),
        },
    }
    write_json(variant_dir / "aggregate_summary.json", aggregate)
    return aggregate


def plot_learning_curves(env_slug: str, output_root: Path) -> None:
    plot_dir = output_root / env_slug / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plotted_any = False
    plt.figure(figsize=(8, 5))
    for variant in VARIANT_NAMES:
        variant_dir = output_root / env_slug / variant
        run_dirs = _latest_seed_runs(variant_dir)
        curves: dict[int, list[float]] = defaultdict(list)
        for run_dir in run_dirs:
            for row in read_jsonl(run_dir / "eval_metrics.jsonl"):
                if row.get("return_mean") is not None:
                    curves[int(row["global_env_step"])].append(float(row["return_mean"]))
        if not curves:
            continue
        steps = sorted(curves)
        means = [float(np.mean(curves[step])) for step in steps]
        plt.plot(steps, means, label=variant)
        plotted_any = True
    if plotted_any:
        plt.xlabel("Environment steps")
        plt.ylabel("Mean evaluation return")
        plt.title("LunarLander-v3 DQN ablations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "eval_learning_curves.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    profile = get_env_profile(args.env)
    output_root = Path(args.output_root)
    written = []
    for variant in VARIANT_NAMES:
        aggregate = aggregate_variant(profile.slug, variant, output_root)
        if aggregate is not None:
            written.append(variant)
    plot_learning_curves(profile.slug, output_root)
    if written:
        print(f"Aggregated variants: {', '.join(written)}")
    else:
        print(f"No completed runs found under {output_root / profile.slug}")


if __name__ == "__main__":
    main()
