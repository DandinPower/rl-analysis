from rl_analysis.aggregate import aggregate_variant
from rl_analysis.utils import write_json


def test_aggregate_variant_writes_summary(tmp_path):
    run_dir = tmp_path / "lunarlander_v3" / "dqn" / "seed_0" / "run"
    run_dir.mkdir(parents=True)
    write_json(
        run_dir / "summary.json",
        {
            "env_id": "LunarLander-v3",
            "variant": "dqn",
            "seed": 0,
            "training_completed": True,
            "performance": {
                "final_eval_return_mean": 10.0,
                "best_eval_return_mean": 12.0,
                "area_under_eval_curve": 100.0,
            },
            "sample_efficiency": {
                "threshold": 200.0,
                "first_step_reaching_threshold": None,
            },
            "stability": {
                "catastrophic_collapse_count": 0,
                "largest_eval_drop": 0.0,
            },
            "failure_diagnostics": {
                "nan_detected": False,
                "q_value_explosion": False,
                "early_terminated": False,
            },
        },
    )
    aggregate = aggregate_variant("lunarlander_v3", "dqn", tmp_path)
    assert aggregate["num_seeds"] == 1
    assert aggregate["final_eval_return"]["mean"] == 10.0
    assert (tmp_path / "lunarlander_v3" / "dqn" / "aggregate_summary.json").exists()


def test_aggregate_variant_includes_freeway_milestones(tmp_path):
    for seed, step in [(0, 100), (1, None)]:
        run_dir = tmp_path / "freeway" / "dqn" / f"seed_{seed}" / "run"
        run_dir.mkdir(parents=True)
        write_json(
            run_dir / "summary.json",
            {
                "env_id": "ALE/Freeway-v5",
                "variant": "dqn",
                "seed": seed,
                "training_completed": True,
                "performance": {
                    "final_eval_return_mean": 10.0 + seed,
                    "best_eval_return_mean": 12.0 + seed,
                    "area_under_eval_curve": 100.0,
                },
                "sample_efficiency": {
                    "threshold": 15.0,
                    "first_step_reaching_threshold": step,
                },
                "stability": {
                    "catastrophic_collapse_count": 0,
                    "largest_eval_drop": 0.0,
                },
                "freeway": {
                    "final_zero_score_rate": 0.5,
                    "score_thresholds": [15.0],
                    "first_step_reaching_score_thresholds": {"15.0": step},
                    "reference_random_score": 0.0,
                    "reference_human_score": 29.6,
                    "reference_dqn_score": 30.3,
                },
                "failure_diagnostics": {
                    "nan_detected": False,
                    "q_value_explosion": False,
                    "early_terminated": False,
                },
            },
        )

    aggregate = aggregate_variant("freeway", "dqn", tmp_path)
    assert aggregate["freeway"]["score_threshold_success_rates"]["15.0"] == 0.5
    assert aggregate["freeway"]["score_threshold_first_steps"]["15.0"]["mean"] == 100.0
