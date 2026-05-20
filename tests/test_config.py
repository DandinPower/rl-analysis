import pytest

from rl_analysis.config import build_run_config, get_env_profile, get_variant_spec


def test_variant_flags_match_ablation_names():
    assert get_variant_spec("dqn").use_target_network
    assert get_variant_spec("dqn").use_replay_buffer
    assert not get_variant_spec("dqn_no_target").use_target_network
    assert not get_variant_spec("dqn_no_replay").use_replay_buffer
    assert get_variant_spec("double_dqn").use_double_dqn
    assert get_variant_spec("dueling_dqn").use_dueling_network
    assert get_variant_spec("double_dueling_dqn").use_double_dqn
    assert get_variant_spec("double_dueling_dqn").use_dueling_network


def test_lunarlander_config_defaults_to_reference_budget(tmp_path):
    config = build_run_config(env_id="LunarLander-v3", variant_name="dqn", seed=0, output_root=tmp_path)
    assert config.env.slug == "lunarlander_v3"
    assert config.training.total_env_steps == 500_000
    assert config.training.eval_frequency_env_steps == 10_000
    assert config.training.checkpoint_frequency_env_steps == 50_000
    assert config.training.num_eval_episodes == 20
    assert config.run_dir.parts[-3:] == ("dqn", "seed_0", config.run_id)


def test_freeway_profile_has_atari_runtime_defaults(tmp_path):
    profile = get_env_profile("freeway")
    assert profile.env_id == "ALE/Freeway-v5"
    assert profile.network_architecture == "cnn"
    assert profile.frame_stack == 4
    assert profile.raw_frame_skip == 1
    assert profile.repeat_action_probability == 0.25
    assert not profile.full_action_space
    assert profile.success_threshold == 15.0
    assert profile.score_thresholds == (5.0, 10.0, 15.0, 22.5)

    config = build_run_config(env_id="freeway", variant_name="dqn", seed=0, output_root=tmp_path)
    assert config.training.total_env_steps == 1_000_000
    assert config.dqn.batch_size == 32
    assert config.dqn.replay_buffer_size == 25_000
    assert config.dqn.learning_starts == 20_000
    assert config.dqn.target_update_frequency_env_steps == 8_000
    assert config.exploration.epsilon_final == 0.01
    assert config.exploration.epsilon_decay_env_steps == 250_000
    assert config.exploration.freeway_up_bias == 0.0
    assert config.optimizer.learning_rate == 6.25e-5
    assert config.optimizer.epsilon == 1.5e-4


def test_freeway_up_bias_override_is_allowed_for_freeway(tmp_path):
    config = build_run_config(
        env_id="freeway",
        variant_name="dqn",
        seed=0,
        output_root=tmp_path,
        exploration_overrides={"freeway_up_bias": 0.5},
    )

    assert config.exploration.freeway_up_bias == 0.5


@pytest.mark.parametrize("freeway_up_bias", [-0.1, 1.1])
def test_freeway_up_bias_rejects_out_of_range_values(tmp_path, freeway_up_bias):
    with pytest.raises(ValueError, match="freeway_up_bias must be between"):
        build_run_config(
            env_id="freeway",
            variant_name="dqn",
            seed=0,
            output_root=tmp_path,
            exploration_overrides={"freeway_up_bias": freeway_up_bias},
        )


def test_freeway_up_bias_rejects_non_freeway_configs(tmp_path):
    with pytest.raises(ValueError, match="freeway_up_bias can only be used"):
        build_run_config(
            env_id="LunarLander-v3",
            variant_name="dqn",
            seed=0,
            output_root=tmp_path,
            exploration_overrides={"freeway_up_bias": 0.5},
        )


def test_no_replay_variant_overrides_replay_hyperparameters(tmp_path):
    config = build_run_config(env_id="LunarLander-v3", variant_name="dqn_no_replay", seed=0, output_root=tmp_path)
    assert config.dqn.batch_size == 1
    assert config.dqn.replay_buffer_size == 0
    assert config.dqn.learning_starts == 0
    assert config.dqn.train_frequency_env_steps == 1
