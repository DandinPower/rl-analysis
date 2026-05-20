import pytest

from rl_analysis import train


def test_selected_seeds_preserve_single_seed_default():
    args = train.parse_args(["--variant", "dqn", "--seed", "7"])

    assert train._selected_seeds(args) == [7]


def test_selected_seeds_default_to_reference_seeds_for_all_variants():
    args = train.parse_args(["--variant", "all"])

    assert train._selected_seeds(args) == [0, 1, 2, 3, 4]


def test_selected_seeds_accept_multiple_seeds_for_single_variant():
    args = train.parse_args(["--variant", "dqn", "--seeds", "2", "3"])

    assert train._selected_seeds(args) == [2, 3]


def test_max_seed_workers_must_be_positive():
    with pytest.raises(SystemExit):
        train.parse_args(["--max-seed-workers", "0"])


def test_freeway_up_bias_is_included_in_exploration_overrides():
    args = train.parse_args(["--env", "freeway", "--freeway-up-bias", "0.75"])

    assert train._exploration_overrides(args)["freeway_up_bias"] == 0.75


def test_main_serial_runs_seed_configs_in_order(monkeypatch, tmp_path):
    seen = []

    def fake_run_config(config):
        seen.append((config.variant.name, config.seed))
        return train.TrainResult(
            run_id=config.run_id,
            run_dir=config.run_dir,
            variant=config.variant.name,
            seed=config.seed,
            final_eval_return_mean=10.0,
        )

    monkeypatch.setattr(train, "_run_config", fake_run_config)

    train.main(
        [
            "--variant",
            "dqn",
            "--seeds",
            "2",
            "3",
            "--serial",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert seen == [("dqn", 2), ("dqn", 3)]


def test_main_parallel_passes_all_seed_configs_and_max_workers(monkeypatch, tmp_path):
    captured = {}

    def fake_run_configs_parallel(configs, max_workers):
        captured["seeds"] = [config.seed for config in configs]
        captured["variants"] = [config.variant.name for config in configs]
        captured["max_workers"] = max_workers
        return 0

    monkeypatch.setattr(train, "_run_configs_parallel", fake_run_configs_parallel)

    train.main(
        [
            "--variant",
            "dqn",
            "--seeds",
            "5",
            "6",
            "--max-seed-workers",
            "2",
            "--output-root",
            str(tmp_path),
        ]
    )

    assert captured == {
        "seeds": [5, 6],
        "variants": ["dqn", "dqn"],
        "max_workers": 2,
    }


def test_main_exits_nonzero_when_any_seed_group_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(train, "_run_configs_parallel", lambda _configs, _max_workers: 1)

    with pytest.raises(SystemExit) as exc_info:
        train.main(
            [
                "--variant",
                "dqn",
                "--seeds",
                "0",
                "1",
                "--output-root",
                str(tmp_path),
            ]
        )

    assert exc_info.value.code == 1
