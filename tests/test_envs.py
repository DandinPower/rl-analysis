import numpy as np

from rl_analysis.envs import (
    FREEWAY_ACTION_DOWN,
    FREEWAY_ACTION_NOOP,
    FREEWAY_ACTION_UP,
    freeway_episode_metrics,
    freeway_up_biased_action_probabilities,
)


def test_freeway_episode_metrics_score_and_action_fractions():
    metrics = freeway_episode_metrics(
        episode_return=12.0,
        episode_length=10,
        action_counts=np.array([3, 6, 1], dtype=np.int64),
        max_score_so_far=18.0,
    )

    assert metrics["score"] == 12
    assert not metrics["zero_score_episode"]
    assert metrics["max_score_so_far"] == 18.0
    assert metrics["action_noop_fraction"] == 0.3
    assert metrics["action_up_fraction"] == 0.6
    assert metrics["action_down_fraction"] == 0.1


def test_freeway_up_biased_action_probabilities_keep_uniform_distribution_at_zero_bias():
    probabilities = freeway_up_biased_action_probabilities(action_dim=3, freeway_up_bias=0.0)

    np.testing.assert_allclose(probabilities, np.array([1 / 3, 1 / 3, 1 / 3]))


def test_freeway_up_biased_action_probabilities_favor_up_action():
    probabilities = freeway_up_biased_action_probabilities(action_dim=3, freeway_up_bias=0.5)

    np.testing.assert_allclose(probabilities[FREEWAY_ACTION_NOOP], 1 / 6)
    np.testing.assert_allclose(probabilities[FREEWAY_ACTION_UP], 2 / 3)
    np.testing.assert_allclose(probabilities[FREEWAY_ACTION_DOWN], 1 / 6)
    np.testing.assert_allclose(probabilities.sum(), 1.0)


def test_freeway_up_biased_action_probabilities_force_up_at_full_bias():
    probabilities = freeway_up_biased_action_probabilities(action_dim=3, freeway_up_bias=1.0)

    np.testing.assert_allclose(probabilities, np.array([0.0, 1.0, 0.0]))
