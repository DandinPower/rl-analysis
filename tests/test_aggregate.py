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
